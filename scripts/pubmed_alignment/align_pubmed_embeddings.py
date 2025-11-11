#!/usr/bin/env python3
"""
Embedding-based PubMed alignment using Sentence Transformers and FAISS.

This script uses semantic embeddings for robust matching of PubMed abstracts
with HuggingFace dataset entries. It's optimized for large-scale processing
(10M+ records) with GPU acceleration support.

Features:
- Sentence transformer embeddings for semantic similarity
- FAISS index for fast nearest neighbor search
- GPU acceleration support
- Incremental processing with caching
- Batch processing for efficiency
- Memory-efficient streaming
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from datasets import load_dataset
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install sentence-transformers faiss-cpu datasets")
    print("For GPU support: pip install faiss-gpu")
    sys.exit(1)

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
    match_type: str = "embedding"


class EmbeddingCache:
    """Cache for computed embeddings to avoid recomputation."""

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self.cache = {}

        if cache_file and cache_file.exists():
            logger.info(f"Loading embedding cache from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def get_hash(self, text: str) -> str:
        """Get hash for text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        text_hash = self.get_hash(text)
        return self.cache.get(text_hash)

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        text_hash = self.get_hash(text)
        self.cache[text_hash] = embedding

    def save(self):
        """Save cache to disk."""
        if self.cache_file:
            logger.info(f"Saving {len(self.cache)} embeddings to cache")
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)


class EmbeddingMatcher:
    """Embedding-based matcher using sentence transformers and FAISS."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        use_gpu: bool = False,
        cache_file: Optional[Path] = None,
        batch_size: int = 32,
    ):
        """
        Initialize embedding matcher.

        Args:
            model_name: Sentence transformer model name
            similarity_threshold: Minimum cosine similarity for matches
            use_gpu: Whether to use GPU acceleration
            cache_file: Path to embedding cache file
            batch_size: Batch size for embedding computation
        """
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        # Load sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        # Initialize cache
        self.cache = EmbeddingCache(cache_file)

        # FAISS index (will be created during indexing)
        self.faiss_index = None
        self.abstract_metadata = []

        # Statistics
        self.stats = {
            "total_abstracts": 0,
            "total_dataset_items": 0,
            "matches_found": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text for consistency."""
        if not text:
            return ""
        return text.strip()

    def compute_embeddings(
        self, texts: List[str], desc: str = "Computing embeddings"
    ) -> np.ndarray:
        """
        Compute embeddings for a list of texts with caching.

        Args:
            texts: List of texts to embed
            desc: Description for progress bar

        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        # Check cache
        for i, text in enumerate(texts):
            cached_emb = self.cache.get(text)
            if cached_emb is not None:
                embeddings.append(cached_emb)
                self.stats["cache_hits"] += 1
            else:
                embeddings.append(None)
                texts_to_compute.append(text)
                indices_to_compute.append(i)
                self.stats["cache_misses"] += 1

        # Compute missing embeddings
        if texts_to_compute:
            logger.info(f"Computing {len(texts_to_compute)} new embeddings")
            new_embeddings = self.model.encode(
                texts_to_compute,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )

            # Update cache and results
            for idx, text, emb in zip(
                indices_to_compute, texts_to_compute, new_embeddings
            ):
                self.cache.set(text, emb)
                embeddings[idx] = emb

        return np.array(embeddings)

    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search.

        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
        """
        logger.info(f"Building FAISS index for {len(embeddings)} embeddings")

        # Ensure embeddings are contiguous and float32
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        # Create index
        if self.use_gpu:
            try:
                # GPU index
                logger.info("Creating GPU FAISS index")
                res = faiss.StandardGpuResources()

                # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
                cpu_index = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception as e:
                logger.warning(f"GPU index failed: {e}. Falling back to CPU.")
                self.use_gpu = False

        if not self.use_gpu:
            # CPU index - use HNSW for faster search
            logger.info("Creating CPU FAISS index with HNSW")
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.faiss_index.hnsw.efConstruction = 40
            self.faiss_index.hnsw.efSearch = 16

        # Add embeddings to index
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def index_abstracts(
        self,
        abstracts_file: Path,
        debug_mode: bool = False,
        max_abstracts: Optional[int] = None,
    ):
        """
        Index abstracts from file.

        Args:
            abstracts_file: Path to abstracts file (one per line)
            debug_mode: If True, limit to first 10k abstracts
            max_abstracts: Maximum number of abstracts to index
        """
        logger.info(f"Reading abstracts from {abstracts_file}")

        abstracts = []

        with open(abstracts_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, desc="Reading abstracts")):
                if debug_mode and line_num >= 10000:
                    break
                if max_abstracts and line_num >= max_abstracts:
                    break

                line = line.strip()
                if not line:
                    continue

                abstracts.append({"line_num": line_num, "text": line})
                self.stats["total_abstracts"] += 1

        logger.info(f"Read {len(abstracts)} abstracts")

        # Compute embeddings
        texts = [a["text"] for a in abstracts]
        embeddings = self.compute_embeddings(texts, desc="Embedding abstracts")

        # Build FAISS index
        self.build_faiss_index(embeddings)

        # Store metadata
        self.abstract_metadata = abstracts

        # Save cache
        self.cache.save()

        logger.info("Indexing complete!")

    def find_match(
        self, pubmed_id: str, mesh_ids: List[str], text: str, k: int = 5
    ) -> Optional[MatchResult]:
        """
        Find best match for a dataset entry.

        Args:
            pubmed_id: PubMed ID
            mesh_ids: MeSH IDs
            text: Query text
            k: Number of nearest neighbors to retrieve

        Returns:
            Best match if similarity >= threshold, else None
        """
        if not text:
            return None

        # Compute query embedding
        query_embedding = self.compute_embeddings([text], desc="")[0]
        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype("float32")
        )

        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, k)

        # Get best match above threshold
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.abstract_metadata):
                continue

            # Convert inner product to cosine similarity (already normalized)
            similarity = float(similarity)

            if similarity >= self.threshold:
                abstract = self.abstract_metadata[idx]
                return MatchResult(
                    pubmed_id=pubmed_id,
                    mesh_ids=mesh_ids,
                    abstract_line=abstract["line_num"],
                    abstract_text=abstract["text"],
                    dataset_text=text,
                    similarity=similarity,
                )

        return None

    def process_dataset(
        self,
        dataset_path: str,
        text_column: str,
        id_column: str,
        mesh_id_column: str,
        output_file: Path,
        save_every: int = 1000,
        max_items: Optional[int] = None,
    ):
        """
        Process HuggingFace dataset and find matches.

        Args:
            dataset_path: HuggingFace dataset path
            text_column: Column name for text
            id_column: Column name for PubMed ID
            mesh_id_column: Column name for MeSH IDs
            output_file: Output file path
            save_every: Save matches every N items
            max_items: Maximum items to process (for testing)
        """
        logger.info(f"Processing dataset: {dataset_path}")

        # Load dataset
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

        # Clear output file
        if output_file.exists():
            output_file.unlink()

        matches = []
        batch_texts = []
        batch_metadata = []

        for item_idx, item in enumerate(tqdm(dataset, desc="Processing dataset")):
            if max_items and item_idx >= max_items:
                break

            try:
                pubmed_id = str(item.get(id_column, ""))
                mesh_ids = list(item.get(mesh_id_column, []))
                text = item.get(text_column, "")

                if not pubmed_id or not text:
                    continue

                self.stats["total_dataset_items"] += 1

                # Batch processing for efficiency
                batch_texts.append(text)
                batch_metadata.append((pubmed_id, mesh_ids))

                # Process batch
                if len(batch_texts) >= self.batch_size:
                    batch_matches = self._process_batch(batch_texts, batch_metadata)
                    matches.extend(batch_matches)
                    batch_texts = []
                    batch_metadata = []

                # Save periodically
                if len(matches) >= save_every:
                    self._save_matches(matches, output_file)
                    matches = []
                    # Save cache periodically
                    self.cache.save()

            except Exception as e:
                logger.warning(f"Error processing item {item_idx}: {e}")
                continue

        # Process remaining batch
        if batch_texts:
            batch_matches = self._process_batch(batch_texts, batch_metadata)
            matches.extend(batch_matches)

        # Save remaining matches
        if matches:
            self._save_matches(matches, output_file)

        # Final cache save
        self.cache.save()

        logger.info("Dataset processing complete!")

    def _process_batch(
        self, texts: List[str], metadata: List[Tuple[str, List[str]]]
    ) -> List[MatchResult]:
        """Process a batch of texts."""
        matches = []

        # Compute embeddings for batch
        embeddings = self.compute_embeddings(texts, desc="")
        embeddings = np.ascontiguousarray(embeddings.astype("float32"))

        # Batch search in FAISS
        similarities, indices = self.faiss_index.search(embeddings, k=5)

        # Process results
        for i, (pubmed_id, mesh_ids) in enumerate(metadata):
            best_similarity = similarities[i][0]
            best_idx = indices[i][0]

            if best_idx >= 0 and best_similarity >= self.threshold:
                abstract = self.abstract_metadata[best_idx]
                match = MatchResult(
                    pubmed_id=pubmed_id,
                    mesh_ids=mesh_ids,
                    abstract_line=abstract["line_num"],
                    abstract_text=abstract["text"],
                    dataset_text=texts[i],
                    similarity=float(best_similarity),
                )
                matches.append(match)
                self.stats["matches_found"] += 1

        return matches

    def _save_matches(self, matches: List[MatchResult], output_file: Path):
        """Save matches to file."""
        with open(output_file, "a", encoding="utf-8") as f:
            for match in matches:
                json.dump(asdict(match), f, ensure_ascii=False)
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based PubMed alignment using sentence transformers and FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (ordered by speed/quality tradeoff):
  all-MiniLM-L6-v2           - Fast, good quality (default)
  all-mpnet-base-v2          - Slower, better quality
  multi-qa-MiniLM-L6-cos-v1  - Optimized for semantic search
  allenai-specter            - Scientific papers (recommended for PubMed)

Examples:
  # Basic usage with default model
  python align_pubmed_embeddings.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids

  # Use scientific paper model with GPU
  python align_pubmed_embeddings.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids \\
      --model allenai-specter \\
      --gpu \\
      --threshold 0.75

  # Debug mode (first 10k abstracts)
  python align_pubmed_embeddings.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids \\
      --debug
        """,
    )

    parser.add_argument(
        "--abstracts",
        type=Path,
        required=True,
        help="Path to abstracts file (one per line)",
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset path"
    )

    parser.add_argument(
        "--text-column", type=str, required=True, help="Text column name in dataset"
    )

    parser.add_argument(
        "--id-column", type=str, required=True, help="PubMed ID column name in dataset"
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
        default="pubmed_matches_embeddings.json",
        help="Output file (default: pubmed_matches_embeddings.json)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32)",
    )

    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")

    parser.add_argument(
        "--cache",
        type=Path,
        default="embeddings_cache.pkl",
        help="Embedding cache file (default: embeddings_cache.pkl)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (process only 10k abstracts)"
    )

    parser.add_argument(
        "--max-abstracts", type=int, help="Maximum number of abstracts to index"
    )

    parser.add_argument(
        "--max-items", type=int, help="Maximum number of dataset items to process"
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

    logger.info("=" * 70)
    logger.info("EMBEDDING-BASED PUBMED ALIGNMENT")
    logger.info("=" * 70)
    logger.info(f"Abstracts file: {args.abstracts}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"GPU acceleration: {args.gpu}")
    logger.info(f"Cache file: {args.cache}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 70)

    try:
        # Initialize matcher
        matcher = EmbeddingMatcher(
            model_name=args.model,
            similarity_threshold=args.threshold,
            use_gpu=args.gpu,
            cache_file=args.cache,
            batch_size=args.batch_size,
        )

        # Index abstracts
        matcher.index_abstracts(
            abstracts_file=args.abstracts,
            debug_mode=args.debug,
            max_abstracts=args.max_abstracts,
        )

        # Process dataset
        matcher.process_dataset(
            dataset_path=args.dataset,
            text_column=args.text_column,
            id_column=args.id_column,
            mesh_id_column=args.mesh_ids_column,
            output_file=args.output,
            max_items=args.max_items,
        )

        # Print statistics
        logger.info("=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total abstracts indexed: {matcher.stats['total_abstracts']:,}")
        logger.info(
            f"Total dataset items processed: {matcher.stats['total_dataset_items']:,}"
        )
        logger.info(f"Total matches found: {matcher.stats['matches_found']:,}")
        logger.info(f"Cache hits: {matcher.stats['cache_hits']:,}")
        logger.info(f"Cache misses: {matcher.stats['cache_misses']:,}")

        if matcher.stats["total_dataset_items"] > 0:
            match_rate = (
                matcher.stats["matches_found"] / matcher.stats["total_dataset_items"]
            ) * 100
            logger.info(f"Match rate: {match_rate:.2f}%")

        cache_hit_rate = (
            matcher.stats["cache_hits"]
            / (matcher.stats["cache_hits"] + matcher.stats["cache_misses"])
            * 100
        )
        logger.info(f"Cache hit rate: {cache_hit_rate:.2f}%")

        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
