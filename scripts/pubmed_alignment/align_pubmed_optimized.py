#!/usr/bin/env python3
"""
Optimized script to align PubMed IDs with their corresponding text using efficient indexing strategies.

This script uses multiple optimization techniques to handle large datasets (10M+ records):
1. Locality Sensitive Hashing (LSH) with MinHash for fast approximate matching
2. TF-IDF based blocking for reducing search space
3. Multi-stage filtering (coarse to fine)
4. Efficient similarity computation with early stopping

Strategies:
- exact_hash: Fast exact matching using MD5 hashes
- lsh: Approximate matching using MinHash LSH
- tfidf_blocking: Block by TF-IDF similarity, then fine-grained matching
- hybrid: Combination of all strategies
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional, Set
import re
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import math
from tqdm import tqdm
import pickle

try:
    from datasets import load_dataset
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    import numpy as np
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install datasets nltk numpy")
    sys.exit(1)

# Download NLTK data if needed
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        print(f"Downloading NLTK {resource}...")
        nltk.download(resource)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a matching operation."""

    pubmed_id: str
    mesh_ids: List[str]
    abstract_line: int
    abstract_text: str
    dataset_text: str
    similarity: float
    match_type: str
    matched_sentence: Optional[str] = None


# ==================== Text Processing ====================


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def extract_sentences(text: str, max_sentences: int = 3) -> List[str]:
    """Extract normalized sentences from text."""
    if not text:
        return []
    sentences = sent_tokenize(text)
    return [normalize_text(sent) for sent in sentences[:max_sentences] if sent.strip()]


def get_word_tokens(text: str) -> List[str]:
    """Get word tokens from text."""
    return [w for w in word_tokenize(text.lower()) if w.isalnum()]


def create_text_hash(text: str) -> str:
    """Create MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def create_shingles(text: str, k: int = 3) -> Set[str]:
    """Create k-shingles (k-grams) from text."""
    words = get_word_tokens(text)
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ==================== MinHash LSH ====================


class MinHashLSH:
    """MinHash-based Locality Sensitive Hashing for fast approximate matching."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.5):
        """
        Initialize MinHash LSH.

        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate)
            threshold: Jaccard similarity threshold for matches
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.bands = self._optimal_bands(num_perm, threshold)
        self.rows = num_perm // self.bands
        self.hashtables = [defaultdict(list) for _ in range(self.bands)]
        self.keys = {}
        self.data = {}

    def _optimal_bands(self, num_perm: int, threshold: float) -> int:
        """Calculate optimal number of bands for given threshold."""
        # bands * rows = num_perm
        # threshold ≈ (1/bands)^(1/rows)
        best_bands = 1
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                t = (1.0 / b) ** (1.0 / r)
                if abs(t - threshold) < abs(
                    (1.0 / best_bands) ** (1.0 / (num_perm // best_bands)) - threshold
                ):
                    best_bands = b
        return best_bands

    def _create_minhash(self, shingles: Set[str]) -> List[int]:
        """Create MinHash signature for a set of shingles."""
        if not shingles:
            return [0] * self.num_perm

        # Use simple hash functions: h_i(x) = (a_i * hash(x) + b_i) mod c
        signature = []
        for i in range(self.num_perm):
            a = (i * 7 + 13) % 1000000007
            b = (i * 11 + 17) % 1000000007
            min_hash = float("inf")

            for shingle in shingles:
                h = hash(shingle)
                hash_val = (a * h + b) % 1000000007
                min_hash = min(min_hash, hash_val)

            signature.append(min_hash)

        return signature

    def insert(self, key: str, text: str, metadata: Dict):
        """Insert a text document into the LSH index."""
        shingles = create_shingles(text)
        minhash = self._create_minhash(shingles)

        # Store data
        self.keys[key] = minhash
        self.data[key] = metadata

        # Insert into hash tables (one per band)
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = tuple(minhash[start:end])
            self.hashtables[band_idx][band_hash].append(key)

    def query(self, text: str, top_k: int = 10) -> List[Tuple[str, Dict, float]]:
        """Query for similar documents."""
        shingles = create_shingles(text)
        query_minhash = self._create_minhash(shingles)

        # Find candidate matches
        candidates = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = tuple(query_minhash[start:end])
            candidates.update(self.hashtables[band_idx].get(band_hash, []))

        # Calculate actual Jaccard similarity for candidates
        results = []
        for candidate_key in candidates:
            candidate_minhash = self.keys[candidate_key]
            # Estimate Jaccard similarity from MinHash
            similarity = (
                sum(
                    1
                    for i in range(self.num_perm)
                    if query_minhash[i] == candidate_minhash[i]
                )
                / self.num_perm
            )

            if similarity >= self.threshold:
                results.append((candidate_key, self.data[candidate_key], similarity))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# ==================== TF-IDF Blocking ====================


class TFIDFBlocker:
    """TF-IDF based blocking to reduce search space."""

    def __init__(self, num_blocks: int = 1000):
        """
        Initialize TF-IDF blocker.

        Args:
            num_blocks: Number of blocks to partition documents into
        """
        self.num_blocks = num_blocks
        self.idf = {}
        self.blocks = defaultdict(list)
        self.doc_count = 0

    def _compute_tf(self, text: str) -> Counter:
        """Compute term frequency."""
        words = get_word_tokens(text)
        return Counter(words)

    def _compute_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for text."""
        tf = self._compute_tf(text)
        tfidf = {}

        for word, freq in tf.items():
            if word in self.idf:
                tfidf[word] = freq * self.idf[word]

        # Normalize
        norm = math.sqrt(sum(v**2 for v in tfidf.values()))
        if norm > 0:
            tfidf = {k: v / norm for k, v in tfidf.items()}

        return tfidf

    def _get_block_id(self, text: str) -> int:
        """Assign document to a block based on TF-IDF."""
        tfidf = self._compute_tfidf_vector(text)
        if not tfidf:
            return 0

        # Use hash of top terms to assign block
        top_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:5]
        block_hash = hash(tuple(t[0] for t in top_terms))
        return block_hash % self.num_blocks

    def build_idf(self, texts: List[str]):
        """Build IDF from a collection of texts."""
        logger.info("Building IDF index...")
        doc_freq = Counter()

        for text in tqdm(texts, desc="Computing IDF"):
            words = set(get_word_tokens(text))
            doc_freq.update(words)
            self.doc_count += 1

        # Compute IDF
        for word, freq in doc_freq.items():
            self.idf[word] = math.log(self.doc_count / (1 + freq))

        logger.info(f"IDF index built with {len(self.idf)} terms")

    def add_document(self, doc_id: str, text: str, metadata: Dict):
        """Add document to blocker."""
        block_id = self._get_block_id(text)
        self.blocks[block_id].append((doc_id, text, metadata))

    def get_candidate_blocks(self, text: str, num_blocks: int = 3) -> List[int]:
        """Get candidate blocks for a query."""
        block_id = self._get_block_id(text)
        # Return primary block and neighboring blocks
        candidates = [block_id]
        for offset in range(1, num_blocks):
            candidates.append((block_id + offset) % self.num_blocks)
            candidates.append((block_id - offset) % self.num_blocks)
        return candidates[:num_blocks]

    def query(self, text: str) -> List[Tuple[str, str, Dict]]:
        """Get candidate documents from relevant blocks."""
        candidate_blocks = self.get_candidate_blocks(text)
        results = []

        for block_id in candidate_blocks:
            results.extend(self.blocks[block_id])

        return results


# ==================== Length-based Filtering ====================


class LengthFilter:
    """Filter candidates based on text length."""

    def __init__(self, tolerance: float = 0.3):
        """
        Initialize length filter.

        Args:
            tolerance: Relative length difference tolerance (0.3 = 30%)
        """
        self.tolerance = tolerance
        self.index = defaultdict(list)

    def add(self, key: str, text: str, metadata: Dict):
        """Add document to index."""
        length_bucket = len(text) // 100  # Bucket by length
        self.index[length_bucket].append((key, text, metadata))

    def query(self, text: str) -> List[Tuple[str, str, Dict]]:
        """Get candidates with similar length."""
        query_len = len(text)
        length_bucket = query_len // 100

        # Get candidates from nearby buckets
        min_len = int(query_len * (1 - self.tolerance))
        max_len = int(query_len * (1 + self.tolerance))

        min_bucket = min_len // 100
        max_bucket = max_len // 100

        candidates = []
        for bucket in range(min_bucket, max_bucket + 1):
            for key, text, metadata in self.index[bucket]:
                if min_len <= len(text) <= max_len:
                    candidates.append((key, text, metadata))

        return candidates


# ==================== Exact Hash Matching ====================


class ExactHashIndex:
    """Fast exact matching using hash index."""

    def __init__(self):
        self.index = defaultdict(list)

    def add(self, text: str, metadata: Dict):
        """Add text to index."""
        text_hash = create_text_hash(normalize_text(text))
        self.index[text_hash].append(metadata)

    def query(self, text: str) -> List[Dict]:
        """Query for exact matches."""
        text_hash = create_text_hash(normalize_text(text))
        return self.index.get(text_hash, [])


# ==================== Main Matching Pipeline ====================


class AbstractMatcher:
    """Main class for matching abstracts with dataset entries."""

    def __init__(self, strategy: str = "hybrid", similarity_threshold: float = 0.7):
        """
        Initialize matcher.

        Args:
            strategy: Matching strategy (exact_hash, lsh, tfidf_blocking, hybrid)
            similarity_threshold: Minimum similarity for matches
        """
        self.strategy = strategy
        self.threshold = similarity_threshold

        # Initialize indexes based on strategy
        self.exact_index = ExactHashIndex()

        if strategy in ["lsh", "hybrid"]:
            self.lsh_index = MinHashLSH(num_perm=128, threshold=similarity_threshold)

        if strategy in ["tfidf_blocking", "hybrid"]:
            self.tfidf_blocker = TFIDFBlocker(num_blocks=5)
            self.length_filter = LengthFilter(tolerance=0.1)

        self.stats = {
            "total_abstracts": 0,
            "total_dataset_items": 0,
            "matches_found": 0,
            "exact_matches": 0,
            "lsh_matches": 0,
            "tfidf_matches": 0,
        }

    def index_abstracts(
        self, abstracts_file: Path, chunk_size: int = 10000, debug_mode: bool = False
    ):
        """Index abstracts from file."""
        logger.info(f"Indexing abstracts from {abstracts_file}")

        abstracts_data = []

        with open(abstracts_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, desc="Reading abstracts")):
                if debug_mode and line_num >= 10000:
                    break

                line = line.strip()
                if not line:
                    continue

                sentences = extract_sentences(line)
                if not sentences:
                    continue

                self.stats["total_abstracts"] += 1

                # Store abstract data
                metadata = {
                    "line_num": line_num,
                    "text": line,
                    "sentences": sentences,
                }
                abstracts_data.append((line, metadata))

                # Index each sentence for exact matching
                for sent in sentences:
                    self.exact_index.add(sent, metadata)

        logger.info(f"Indexed {self.stats['total_abstracts']} abstracts")

        # Build strategy-specific indexes
        if self.strategy in ["lsh", "hybrid"]:
            logger.info("Building LSH index...")
            for text, metadata in tqdm(abstracts_data, desc="LSH indexing"):
                key = f"abs_{metadata['line_num']}"
                self.lsh_index.insert(key, text, metadata)

        if self.strategy in ["tfidf_blocking", "hybrid"]:
            logger.info("Building TF-IDF blocker...")
            texts = [text for text, _ in abstracts_data]
            self.tfidf_blocker.build_idf(texts)

            for text, metadata in tqdm(abstracts_data, desc="TF-IDF blocking"):
                key = f"abs_{metadata['line_num']}"
                self.tfidf_blocker.add_document(key, text, metadata)
                self.length_filter.add(key, text, metadata)

        logger.info("Indexing complete!")

    def find_match(
        self, pubmed_id: str, mesh_ids: List[str], text: str
    ) -> Optional[MatchResult]:
        """Find best match for a dataset entry."""
        sentences = extract_sentences(text)

        best_match = None
        best_similarity = 0.0

        # Strategy 1: Exact hash matching (fastest)
        for sent in sentences:
            exact_matches = self.exact_index.query(sent)
            if exact_matches:
                match = exact_matches[0]
                return MatchResult(
                    pubmed_id=pubmed_id,
                    mesh_ids=mesh_ids,
                    abstract_line=match["line_num"],
                    abstract_text=match["text"],
                    dataset_text=text,
                    similarity=1.0,
                    match_type="exact_hash",
                    matched_sentence=sent,
                )

        # Strategy 2: LSH approximate matching
        if self.strategy in ["lsh", "hybrid"] and best_similarity < self.threshold:
            lsh_results = self.lsh_index.query(text, top_k=5)
            for key, metadata, similarity in lsh_results:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = MatchResult(
                        pubmed_id=pubmed_id,
                        mesh_ids=mesh_ids,
                        abstract_line=metadata["line_num"],
                        abstract_text=metadata["text"],
                        dataset_text=text,
                        similarity=similarity,
                        match_type="lsh",
                    )

        # Strategy 3: TF-IDF blocking with fine-grained matching
        if (
            self.strategy in ["tfidf_blocking", "hybrid"]
            and best_similarity < self.threshold
        ):
            # First filter by length
            length_candidates = self.length_filter.query(text)

            if length_candidates:
                # Compute actual similarity for length-filtered candidates
                query_words = set(get_word_tokens(text))

                for key, candidate_text, metadata in length_candidates[
                    :100
                ]:  # Limit candidates
                    candidate_words = set(get_word_tokens(candidate_text))

                    # Jaccard similarity
                    if len(query_words) == 0 or len(candidate_words) == 0:
                        continue

                    intersection = len(query_words & candidate_words)
                    union = len(query_words | candidate_words)
                    similarity = intersection / union if union > 0 else 0.0

                    if similarity > best_similarity and similarity >= self.threshold:
                        best_similarity = similarity
                        best_match = MatchResult(
                            pubmed_id=pubmed_id,
                            mesh_ids=mesh_ids,
                            abstract_line=metadata["line_num"],
                            abstract_text=metadata["text"],
                            dataset_text=text,
                            similarity=similarity,
                            match_type="tfidf_blocking",
                        )

        return best_match

    def process_dataset(
        self,
        dataset_path: str,
        text_column: str,
        id_column: str,
        mesh_id_column: str,
        output_file: Path,
        chunk_size: int = 10_000,
        save_every: int = 1000,
    ):
        """Process HuggingFace dataset and find matches."""
        logger.info(f"Processing dataset {dataset_path}")

        matches = []

        try:
            dataset = load_dataset(dataset_path, streaming=True)
            if isinstance(dataset, dict):
                for split_name in ["train", "validation", "test", "all"]:
                    if split_name in dataset:
                        dataset = dataset[split_name]
                        break
                else:
                    dataset = list(dataset.values())[0]
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

        # Clear output file if it exists
        if output_file.exists():
            output_file.unlink()

        for item in tqdm(dataset, desc="Matching dataset"):
            try:
                pubmed_id = str(item.get(id_column, ""))
                mesh_ids = list(item.get(mesh_id_column, []))
                text = item.get(text_column, "")

                if not pubmed_id or not text:
                    continue

                self.stats["total_dataset_items"] += 1

                # Find match
                match = self.find_match(pubmed_id, mesh_ids, text)

                if match:
                    matches.append(match)
                    self.stats["matches_found"] += 1

                    if match.match_type == "exact_hash":
                        self.stats["exact_matches"] += 1
                    elif match.match_type == "lsh":
                        self.stats["lsh_matches"] += 1
                    elif match.match_type == "tfidf_blocking":
                        self.stats["tfidf_matches"] += 1

                # Save periodically
                if len(matches) >= save_every:
                    self._save_matches(matches, output_file)
                    matches = []

            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue

        # Save remaining matches
        if matches:
            self._save_matches(matches, output_file)

        logger.info("Dataset processing complete!")

    def _save_matches(self, matches: List[MatchResult], output_file: Path):
        """Save matches to file."""
        with open(output_file, "a", encoding="utf-8") as f:
            for match in matches:
                json.dump(asdict(match), f, ensure_ascii=False)
                f.write("\n")


# ==================== CLI ====================


def main():
    parser = argparse.ArgumentParser(
        description="Optimized PubMed alignment with LSH and blocking strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Matching Strategies:
  exact_hash      - Fast exact matching using MD5 hashes (fastest, least memory)
  lsh             - Approximate matching with MinHash LSH (balanced)
  tfidf_blocking  - TF-IDF blocking with Jaccard similarity (good for fuzzy matches)
  hybrid          - Combines all strategies (recommended, most accurate)

Examples:
  # Fast exact matching only
  python align_pubmed_optimized.py --abstracts abstracts.txt --dataset my_dataset \\
      --text-column text --id-column pmid --mesh-ids-column mesh_ids \\
      --strategy exact_hash

  # LSH approximate matching
  python align_pubmed_optimized.py --abstracts abstracts.txt --dataset my_dataset \\
      --text-column text --id-column pmid --mesh-ids-column mesh_ids \\
      --strategy lsh --threshold 0.7

  # Hybrid approach (recommended)
  python align_pubmed_optimized.py --abstracts abstracts.txt --dataset my_dataset \\
      --text-column text --id-column pmid --mesh-ids-column mesh_ids \\
      --strategy hybrid --threshold 0.7
  #
  # python .\align_pubmed_optimized.py --abstracts=T:\pubmed_english_abstracts\pmc_part_002.txt \\
  # --dataset=UMCU\pubmedabstracts_2025 --id-column=meta_pmid \\
  # --mesh-ids-columns=meta_mesh_ids --text-column=text --output=T:\bla.txt \\
  # --strategy hybrid --threshold 0.7
        """,
    )

    parser.add_argument(
        "--abstracts",
        type=Path,
        required=True,
        help="Path to abstracts file (one per line)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset path",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Text column name in dataset",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        required=True,
        help="PubMed ID column name in dataset",
    )

    parser.add_argument(
        "--mesh-ids-column",
        type=str,
        required=True,
        help="MeSH IDs column name in dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="pubmed_matches_optimized.json",
        help="Output file (default: pubmed_matches_optimized.json)",
    )

    parser.add_argument(
        "--strategy",
        choices=["exact_hash", "lsh", "tfidf_blocking", "hybrid"],
        default="hybrid",
        help="Matching strategy (default: hybrid)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Processing chunk size (default: 10000)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (process only first 10k abstracts)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate inputs
    if not args.abstracts.exists():
        logger.error(f"Abstracts file not found: {args.abstracts}")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1:
        logger.error("Threshold must be between 0 and 1")
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("OPTIMIZED PUBMED ALIGNMENT")
    logger.info("=" * 60)
    logger.info(f"Abstracts file: {args.abstracts}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    try:
        # Initialize matcher
        matcher = AbstractMatcher(
            strategy=args.strategy, similarity_threshold=args.threshold
        )

        # Index abstracts
        matcher.index_abstracts(
            args.abstracts, chunk_size=args.chunk_size, debug_mode=args.debug
        )

        # Process dataset
        matcher.process_dataset(
            dataset_path=args.dataset,
            text_column=args.text_column,
            id_column=args.id_column,
            mesh_id_column=args.mesh_ids_column,
            output_file=args.output,
            chunk_size=args.chunk_size,
        )

        # Print statistics
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total abstracts indexed: {matcher.stats['total_abstracts']:,}")
        logger.info(
            f"Total dataset items processed: {matcher.stats['total_dataset_items']:,}"
        )
        logger.info(f"Total matches found: {matcher.stats['matches_found']:,}")
        logger.info(f"  - Exact hash matches: {matcher.stats['exact_matches']:,}")
        logger.info(f"  - LSH matches: {matcher.stats['lsh_matches']:,}")
        logger.info(f"  - TF-IDF blocking matches: {matcher.stats['tfidf_matches']:,}")

        if matcher.stats["total_dataset_items"] > 0:
            match_rate = (
                matcher.stats["matches_found"] / matcher.stats["total_dataset_items"]
            ) * 100
            logger.info(f"Match rate: {match_rate:.2f}%")

        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
