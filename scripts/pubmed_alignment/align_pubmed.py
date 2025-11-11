#!/usr/bin/env python3
"""
Script to align PubMed IDs with their corresponding text using out-of-core processing.

This script matches abstracts from a text file with PubMed IDs from a HuggingFace dataset
by finding sentence-level matches rather than full text matches. It's designed to handle
large datasets that don't fit in memory by processing them in chunks.
"""

import argparse
import logging
import json
from math import log
import sys
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Optional
import re
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install datasets nltk")
    sys.exit(1)

# Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab/english")
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt_tab")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBUG_LIMIT = 100_000


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and punctuation."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())
    # Remove common formatting artifacts
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def extract_sentences(text: str, max_sentences: int = 2) -> List[str]:
    """Extract up to max_sentences from the beginning of the text."""
    if not text:
        return []

    sentences = sent_tokenize(text)
    return [normalize_text(sent) for sent in sentences[:max_sentences] if sent.strip()]


def compute_similarity(text1: str, text2: str, threshold: float = 0.8) -> float:
    """Compute similarity between two text strings using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


def create_text_hash(text: str) -> str:
    """Create a hash for quick text comparison."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def read_abstracts_file(
    file_path: Path, chunk_size: int = 1000, debug_mode: bool = False
) -> Iterator[Tuple[int, str, List[str]]]:
    """
    Read abstracts from file in chunks.

    Yields:
        Tuple of (line_number, original_text, normalized_sentences)
    """
    logger.info(f"Reading abstracts from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunk = []
        line_num = 0
        abstracts_processed = 0

        for line in tqdm(f, desc="Reading abstracts"):
            # In debug mode, limit to 100 abstracts
            if debug_mode and abstracts_processed >= DEBUG_LIMIT:
                logger.info(f"Debug mode: Limited to {abstracts_processed} abstracts")
                break

            line = line.strip()
            if line:
                sentences = extract_sentences(line)
                if sentences:  # Only include if we have valid sentences
                    chunk.append((line_num, line, sentences))
                    abstracts_processed += 1
                line_num += 1

                if len(chunk) >= chunk_size:
                    yield from chunk
                    chunk = []

        # Yield remaining items
        if chunk:
            yield from chunk


def process_hf_dataset(
    dataset_path: str,
    text_column: str,
    id_column: str,
    mesh_id_column: str,
    chunk_size: int = 10_000,
) -> Iterator[Tuple[str, str, List[str]]]:
    """
    Process HuggingFace dataset in chunks.

    Yields:
        Tuple of (pubmed_id, original_text, normalized_sentences)
    """
    logger.info(f"Loading HuggingFace dataset from {dataset_path}")

    try:
        dataset = load_dataset(dataset_path, streaming=True)
        # Handle different dataset splits
        if isinstance(dataset, dict):
            # Try common split names
            for split_name in ["train", "validation", "test", "all"]:
                if split_name in dataset:
                    dataset = dataset[split_name]
                    break
            else:
                # Use the first available split
                dataset = list(dataset.values())[0]
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        return

    chunk = []
    processed_count = 0

    for item in dataset:
        try:
            pubmed_id = str(item.get(id_column, ""))
            mesh_ids = list(item.get(mesh_id_column, []))
            text = item.get(text_column, "")

            if pubmed_id and text:
                sentences = extract_sentences(text)
                if sentences:
                    chunk.append((pubmed_id, mesh_ids, text, sentences))

            processed_count += 1

            if len(chunk) >= chunk_size:
                yield from chunk
                chunk = []
                logger.info(f"Processed {processed_count} dataset items")

        except Exception as e:
            logger.warning(f"Error processing item {processed_count}: {e}")
            continue

    # Yield remaining items
    if chunk:
        yield from chunk

    logger.info(f"Finished processing {processed_count} dataset items")


def find_matches(
    abstracts_file: Path,
    dataset_path: str,
    text_column: str,
    id_column: str,
    mesh_id_column: str,
    similarity_threshold: float,
    output_file: Path,
    chunk_size: int,
    debug_mode: bool = False,
) -> Dict[str, int]:
    """
    Find matches between abstracts file and HuggingFace dataset.

    Returns:
        Dictionary with match statistics
    """
    matches = []
    stats = {
        "total_abstracts": 0,
        "total_dataset_items": 0,
        "matches_found": 0,
        "exact_matches": 0,
        "fuzzy_matches": 0,
    }

    # Create index of abstract sentences for faster matching
    logger.info("Building abstract index...")
    abstract_index = {}  # hash -> (line_num, original_text, sentences)

    for line_num, original_text, sentences in read_abstracts_file(
        abstracts_file, chunk_size, debug_mode
    ):
        stats["total_abstracts"] += 1

        for sent in sentences:
            if sent:
                text_hash = create_text_hash(sent)
                if text_hash not in abstract_index:
                    abstract_index[text_hash] = []
                abstract_index[text_hash].append((line_num, original_text, sentences))

    logger.info(
        f"Built index with {len(abstract_index)} unique sentence hashes from {stats['total_abstracts']} abstracts"
    )

    NUM_ABSTRACTS = len(abstract_index)
    # Process dataset and find matches
    logger.info("Finding matches in dataset...")

    similarities = []
    for pubmed_id, mesh_ids, dataset_text, dataset_sentences in tqdm(
        process_hf_dataset(
            dataset_path, text_column, id_column, mesh_id_column, chunk_size
        ),
        desc="Processing dataset",
    ):
        stats["total_dataset_items"] += 1

        best_match = None
        best_similarity = 0.0
        match_type = None
        matched_sentence = None

        # Check each sentence from the dataset against abstract index
        for dataset_sent in dataset_sentences[
            :
        ]:  # Use slice copy to allow modification during iteration
            if not dataset_sent:
                continue

            dataset_hash = create_text_hash(dataset_sent)

            # Check for exact hash match first
            if dataset_hash in abstract_index:
                for line_num, abs_text, abs_sentences in abstract_index[dataset_hash]:
                    best_match = {
                        "pubmed_id": pubmed_id,
                        "mesh_ids": mesh_ids,
                        "abstract_line": line_num,
                        "abstract_text": abs_text,
                        "dataset_text": dataset_text,
                        "matched_sentence": dataset_sent,
                        "similarity": 1.0,
                    }
                    match_type = "exact"
                    best_similarity = 1.0
                    matched_sentence = dataset_sent
                    break

            if (
                best_similarity >= 1.0
            ):  # Exact match found, stop checking more sentences
                break

            # If no exact match, try fuzzy matching against a sample
            if best_similarity < similarity_threshold:
                # Sample some abstract sentences for fuzzy matching to avoid O(n²) complexity
                sample_hashes = list(abstract_index.keys())[
                    : min(1000, len(abstract_index))
                ]

                for sample_hash in sample_hashes:
                    for line_num, abs_text, abs_sentences in abstract_index[
                        sample_hash
                    ]:
                        for abs_sent in abs_sentences:
                            similarity = compute_similarity(dataset_sent, abs_sent)
                            similarities.append(similarity)
                            if (
                                similarity > best_similarity
                                and similarity >= similarity_threshold
                            ):
                                best_match = {
                                    "pubmed_id": pubmed_id,
                                    "mesh_ids": mesh_ids,
                                    "abstract_line": line_num,
                                    "abstract_text": abs_text,
                                    "dataset_text": dataset_text,
                                    "matched_sentence": dataset_sent,
                                    "matched_abstract_sentence": abs_sent,
                                    "similarity": similarity,
                                }
                                best_similarity = similarity
                                match_type = "fuzzy"
                                matched_sentence = dataset_sent

        mean_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        logger.info(f"Max similarity: {max_similarity}")
        logger.info(f"Mean similarity: {mean_similarity}")

        # Remove the matched sentence from dataset_sentences to avoid re-processing
        if matched_sentence and matched_sentence in dataset_sentences:
            dataset_sentences.remove(matched_sentence)

        if best_match:
            matches.append(best_match)
            stats["matches_found"] += 1

            if match_type == "exact":
                stats["exact_matches"] += 1
            else:
                stats["fuzzy_matches"] += 1

            # Log progress periodically
            if stats["matches_found"] % 100 == 0:
                logger.info(f"Found {stats['matches_found']} matches so far...")

            # Log pruning progress
            logger.debug(
                f"Pruned matched sentence. Remaining sentences: {len(dataset_sentences)}"
            )

        if len(matches) % 1000 == 0:
            logger.info(f"Saved {len(matches)} matches so far...")
            with open(output_file, "a", encoding="utf-8") as f:
                for match in matches:
                    json.dump(match, f, ensure_ascii=False)
                    f.write("\n")
            matches = []  # Clear matches after saving

    # Save matches to output file
    logger.info(f"Saving {len(matches)} matches to {output_file}")

    with open(output_file, "a", encoding="utf-8") as f:
        for match in matches:
            json.dump(match, f, ensure_ascii=False)
            f.write("\n")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Align PubMed IDs with corresponding text using out-of-core processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python align_pubmed.py abstracts.txt my_dataset --text-column text --id-column pmid
  python align_pubmed.py abstracts.txt my_dataset --text-column abstract --id-column pubmed_id --threshold 0.9
  python align_pubmed.py abstracts.txt my_dataset --text-column text --id-column pmid --chunk-size 500 --output matches.json
        """,
    )

    parser.add_argument(
        "--abstracts",
        type=Path,
        help="Path to text file with one abstract per line",
    )

    parser.add_argument("--dataset", type=str, help="HuggingFace dataset path or name")

    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Name of the text column in the HuggingFace dataset",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        required=True,
        help="Name of the PubMed ID column in the HuggingFace dataset",
    )

    parser.add_argument(
        "--mesh-ids-column",
        type=str,
        required=True,
        help="Name of the Mesh ID column in the HuggingFace dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="pubmed_matches.json",
        help="Output JSON file for matches (default: pubmed_matches.json)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy matching (default: 0.8)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Chunk size for out-of-core processing (default: 10000)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limits to 100 abstracts for testing)",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate inputs
    if not args.abstracts_file.exists():
        logger.error(f"Abstracts file not found: {args.abstracts_file}")
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1:
        logger.error("Threshold must be between 0 and 1")
        sys.exit(1)

    if args.chunk_size < 1:
        logger.error("Chunk size must be positive")
        sys.exit(1)

    # Create output directory if it doesn't exist
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting PubMed ID alignment...")
    logger.info(f"Abstracts file: {args.abstracts}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Text column: {args.text_column}")
    logger.info(f"ID column: {args.id_column}")
    logger.info(f"Mesh ID column: {args.mesh_ids_column}")
    logger.info(f"Similarity threshold: {args.threshold}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Output file: {args.output}")

    try:
        stats = find_matches(
            abstracts_file=args.abstracts,
            dataset_path=args.dataset,
            text_column=args.text_column,
            id_column=args.id_column,
            mesh_id_column=args.mesh_ids_column,
            similarity_threshold=args.threshold,
            output_file=args.output,
            chunk_size=args.chunk_size,
            debug_mode=args.debug,
        )

        # Print final statistics
        logger.info("=" * 50)
        logger.info("ALIGNMENT COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total abstracts processed: {stats['total_abstracts']:,}")
        logger.info(f"Total dataset items processed: {stats['total_dataset_items']:,}")
        logger.info(f"Total matches found: {stats['matches_found']:,}")
        logger.info(f"Exact matches: {stats['exact_matches']:,}")
        logger.info(f"Fuzzy matches: {stats['fuzzy_matches']:,}")

        if stats["total_dataset_items"] > 0:
            match_rate = (stats["matches_found"] / stats["total_dataset_items"]) * 100
            logger.info(f"Match rate: {match_rate:.2f}%")

        logger.info(f"Results saved to: {args.output}")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
