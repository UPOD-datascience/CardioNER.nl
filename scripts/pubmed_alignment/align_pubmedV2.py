"""
Efficient alignment of unaligned text file with parquet metadata file.
Strategy: Sort by first sentence, then use Levenshtein distance on second sentences
to match texts efficiently for 30M+ records.
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message="resource_tracker")

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysbd
from Levenshtein import distance as levenshtein_distance
from Levenshtein import ratio as levenshtein_ratio
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextAligner:
    def __init__(
        self,
        alignment_file_path: str,
        unaligned_file_path: str,
        output_dir: str,
        chunk_size: int = 100000,
        search_window: int = 10000,
        sentence_mode: str = "pysbd",
    ):
        """
        Initialize the TextAligner.

        Args:
            alignment_file_path: Path to parquet file with metadata
            unaligned_file_path: Path to .txt file with texts (one per line)
            output_dir: Directory to save output files
            chunk_size: Size of chunks for processing
            search_window: Number of records to look ahead from last match
            sentence_mode: Sentence extraction mode - 'pysbd' or 'simple'
        """
        self.alignment_file_path = Path(alignment_file_path)
        self.unaligned_file_path = Path(unaligned_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.search_window = search_window
        self.sentence_mode = sentence_mode

        # Initialize sentence segmenter only if using pysbd mode
        if self.sentence_mode == "pysbd":
            self.segmenter = pysbd.Segmenter(language="en", clean=False)
        else:
            self.segmenter = None

        # Enable pandas progress bars
        tqdm.pandas()

    def extract_sentences(self, text: str, n: int = 3) -> List[str]:
        """
        Extract first n sentences from text using pysbd or simple splitting.

        Args:
            text: Input text
            n: Number of sentences to extract

        Returns:
            List of sentences (padded with empty strings if fewer than n)
        """
        if pd.isna(text) or text == "":
            return [""] * n

        if self.sentence_mode == "simple":
            # Simple mode: split by period
            sentences = [s.strip() for s in text.split(".") if s.strip()]
        else:
            # pysbd mode: use sophisticated sentence segmenter
            sentences = self.segmenter.segment(text)

        # Pad with empty strings if fewer than n sentences
        while len(sentences) < n:
            sentences.append("")
        return sentences[:n]

    def step1_prepare_unaligned_file(self) -> List[str]:
        """
        Step 1: Load file using pandas read_csv, add IDs, extract sentences, write to parquet.
        Skips processing if output file already exists.

        Returns:
            List of paths to the prepared parquet chunk files
        """
        logger.info("Step 1: Preparing unaligned file...")

        # Check if output file already esxists
        output_path = self.output_dir / f"unaligned_step1_chunk_0.parquet"
        if output_path.exists():
            logger.info(
                f"Step 1 output already exists at {output_path}, skipping to step 2..."
            )
            return [str(output_path)]

        # Count lines in text file for sanity check
        logger.info(f"Counting lines in {self.unaligned_file_path}...")
        with open(self.unaligned_file_path, "r", encoding="latin1") as f:
            num_lines = sum(1 for line in f if line.strip())  # Count non-empty lines
        logger.info(f"Text file has {num_lines} non-empty lines")

        # Read using pandas read_csv
        logger.info(f"Reading {self.unaligned_file_path}...")
        df = pd.read_csv(
            self.unaligned_file_path,
            names=["text"],
            header=None,
            sep="\x01",
            skip_blank_lines=True,
            parse_dates=False,
            doublequote=False,
            encoding="latin1",
        )
        logger.info(
            f"Loaded DataFrame with shape: {df.shape}, columns: {list(df.columns)}"
        )

        # Add ID column with range(0, end)
        df.insert(0, "ID", range(0, len(df)))

        # Extract sentences for pipeline compatibility
        logger.info("Extracting sentences...")
        tqdm.pandas(desc="Extracting sentences")
        sentences = df["text"].progress_apply(lambda x: self.extract_sentences(x, 3))
        df["sentence_1"] = sentences.apply(lambda x: x[0])
        df["sentence_2"] = sentences.apply(lambda x: x[1])
        df["sentence_3"] = sentences.apply(lambda x: x[2])

        # Sanity check: parquet rows should not exceed text file lines
        num_parquet_rows = len(df)
        logger.info(f"Parquet will have {num_parquet_rows} rows")
        if num_parquet_rows > num_lines:
            logger.warning(
                f"SANITY CHECK FAILED: Parquet has {num_parquet_rows} rows "
                f"but text file only has {num_lines} lines!"
            )
        else:
            logger.info(
                f"SANITY CHECK PASSED: Parquet rows ({num_parquet_rows}) <= "
                f"text file lines ({num_lines})"
            )

        # Write to parquet with explicit engine
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved to {output_path}")

        # Explicit cleanup to help with resource management
        del df
        gc.collect()

        return [str(output_path)]

    def step2_prepare_alignment_file(self) -> List[str]:
        """
        Step 2: Extract sentences from alignment file with metadata using streaming read/write to single parquet.

        Returns:
            List containing path to the single prepared parquet file
        """
        logger.info("Step 2: Preparing alignment file with streaming read/write...")

        # Check if output file already exists
        output_path = self.output_dir / "aligned_step2.parquet"
        if output_path.exists():
            logger.info(f"Step 2 output already exists at {output_path}, skipping...")
            return [str(output_path)]

        # Stream read the alignment file in chunks
        logger.info(f"Streaming read from {self.alignment_file_path}...")
        parquet_file = pq.ParquetFile(self.alignment_file_path)

        # Get total number of rows for progress tracking
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"Total rows in alignment file: {total_rows}")

        # Initialize ParquetWriter (will be set after first batch to get schema)
        writer = None
        rows_processed = 0

        # Iterate through batches
        for batch_idx, batch in enumerate(
            tqdm(
                parquet_file.iter_batches(batch_size=self.chunk_size),
                desc="Processing batches",
                total=(total_rows // self.chunk_size) + 1,
            )
        ):
            # Convert batch to pandas DataFrame
            df_batch = batch.to_pandas()
            logger.info(
                f"Loaded batch {batch_idx} with shape: {df_batch.shape}, columns: {list(df_batch.columns)}"
            )

            # Extract sentences
            logger.info(
                f"Extracting sentences for batch {batch_idx} ({len(df_batch)} rows)..."
            )
            tqdm.pandas(desc="Extracting sentences", leave=False)
            sentences = df_batch["text"].progress_apply(
                lambda x: self.extract_sentences(x, 3)
            )
            df_batch["sentence_1"] = sentences.apply(lambda x: x[0])
            df_batch["sentence_2"] = sentences.apply(lambda x: x[1])
            df_batch["sentence_3"] = sentences.apply(lambda x: x[2])

            # Keep only necessary columns
            columns_to_keep = [
                "meta_pmid",
                "meta_language",
                "meta_mesh_ids",
                "meta_mesh_terms",
                "meta_pubdate_year",
                "sentence_1",
                "sentence_2",
                "sentence_3",
            ]
            df_batch = df_batch[columns_to_keep]

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df_batch)

            # Initialize writer on first batch
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
                logger.info(f"Initialized ParquetWriter for {output_path}")

            # Write this batch to the file
            writer.write_table(table)
            rows_processed += len(df_batch)

            # Explicit cleanup
            del df_batch
            del sentences
            del table
            gc.collect()

        # Close the writer
        if writer is not None:
            writer.close()
            logger.info(f"Closed ParquetWriter. Total rows written: {rows_processed}")
        else:
            logger.warning("No data was processed!")
            return []

        logger.info(f"Saved single parquet file to {output_path}")
        return [str(output_path)]

    def step3_sort_by_first_sentence(
        self, unaligned_paths: List[str], aligned_paths: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Step 3: Sort both sets of dataframes by first sentence using DuckDB.

        Args:
            unaligned_paths: List of paths to unaligned parquet chunks
            aligned_paths: List of paths to aligned parquet chunks

        Returns:
            Tuple of (sorted_unaligned_paths, sorted_aligned_paths)
        """
        logger.info("Step 3: Sorting by first sentence using DuckDB...")

        # Check if output files already exist
        sorted_unaligned_paths = []
        sorted_aligned_paths = []

        # Check for existing unaligned sorted files
        for chunk_idx in range(len(unaligned_paths)):
            sorted_path = self.output_dir / f"unaligned_step3_chunk_{chunk_idx}.parquet"
            sorted_unaligned_paths.append(str(sorted_path))

        # Check for existing aligned sorted files
        for chunk_idx in range(len(aligned_paths)):
            sorted_path = self.output_dir / f"aligned_step3_chunk_{chunk_idx}.parquet"
            sorted_aligned_paths.append(str(sorted_path))

        # If all output files exist, skip processing
        all_exist = all(
            Path(p).exists() for p in sorted_unaligned_paths + sorted_aligned_paths
        )
        if all_exist:
            logger.info(
                f"Step 3 output already exists ({len(sorted_unaligned_paths)} unaligned, {len(sorted_aligned_paths)} aligned chunks), skipping..."
            )
            return sorted_unaligned_paths, sorted_aligned_paths

        # Reset paths lists for actual processing
        sorted_unaligned_paths = []
        sorted_aligned_paths = []

        conn = duckdb.connect()

        # Sort unaligned chunks
        logger.info("Sorting unaligned chunks...")
        for chunk_idx, chunk_path in enumerate(
            tqdm(unaligned_paths, desc="Sorting unaligned chunks")
        ):
            sorted_path = self.output_dir / f"unaligned_step3_chunk_{chunk_idx}.parquet"
            conn.execute(f"""
                COPY (
                    SELECT * FROM read_parquet('{chunk_path}')
                    ORDER BY sentence_1
                ) TO '{sorted_path}' (FORMAT PARQUET)
            """)
            sorted_unaligned_paths.append(str(sorted_path))

        # Sort aligned chunks
        logger.info("Sorting aligned chunks...")
        for chunk_idx, chunk_path in enumerate(
            tqdm(aligned_paths, desc="Sorting aligned chunks")
        ):
            sorted_path = self.output_dir / f"aligned_step3_chunk_{chunk_idx}.parquet"
            conn.execute(f"""
                COPY (
                    SELECT * FROM read_parquet('{chunk_path}')
                    ORDER BY sentence_1
                ) TO '{sorted_path}' (FORMAT PARQUET)
            """)
            sorted_aligned_paths.append(str(sorted_path))

        conn.close()
        logger.info("Sorting complete")

        return sorted_unaligned_paths, sorted_aligned_paths

    def step4_align_using_levenshtein(
        self,
        sorted_unaligned_paths: List[str],
        sorted_aligned_paths: List[str],
        max_distance: int = 10,
    ) -> str:
        """
        Step 4: Align texts using Levenshtein distance on second sentences.
        Uses iterative reading and writing for memory efficiency.

        Performs metadata scanning before loading to:
        - Check file shapes and column names without loading data
        - Validate required columns exist
        - Estimate memory usage (only for aligned data, unaligned is streamed)

        Reads unaligned data iteratively in batches and writes results incrementally.
        Only loads aligned data fully (sorted for efficient searching).

        Args:
            sorted_unaligned_paths: List of paths to sorted unaligned parquet chunks
            sorted_aligned_paths: List of paths to sorted aligned parquet chunks
            max_distance: Maximum Levenshtein distance to consider a match

        Returns:
            Path to the output file with aligned metadata
        """
        logger.info("Step 4: Aligning using Levenshtein distance...")

        # Check if output file already exists
        output_path = self.output_dir / "alignment_results.parquet"
        if output_path.exists():
            logger.info(f"Step 4 output already exists at {output_path}, skipping...")
            return str(output_path)

        # Metadata scan before loading
        logger.info("Scanning metadata of unaligned files (will be streamed)...")
        total_unaligned_rows = 0
        required_unaligned_cols = {"ID", "sentence_1", "sentence_2"}
        for path in sorted_unaligned_paths:
            parquet_file = pq.ParquetFile(path)
            num_rows = parquet_file.metadata.num_rows
            schema = parquet_file.schema_arrow
            column_names = [field.name for field in schema]
            total_unaligned_rows += num_rows
            logger.info(
                f"  {Path(path).name}: {num_rows:,} rows, columns: {column_names}"
            )

            # Validate required columns
            missing_cols = required_unaligned_cols - set(column_names)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {Path(path).name}: {missing_cols}. "
                    f"Found columns: {column_names}"
                )
        logger.info(
            f"Total unaligned rows to process (streamed): {total_unaligned_rows:,}"
        )

        logger.info("Scanning metadata of aligned files...")
        total_aligned_rows = 0
        required_aligned_cols = {"sentence_1", "sentence_2", "meta_pmid"}
        for path in sorted_aligned_paths:
            parquet_file = pq.ParquetFile(path)
            num_rows = parquet_file.metadata.num_rows
            schema = parquet_file.schema_arrow
            column_names = [field.name for field in schema]
            total_aligned_rows += num_rows
            logger.info(
                f"  {Path(path).name}: {num_rows:,} rows, columns: {column_names}"
            )

            # Validate required columns
            missing_cols = required_aligned_cols - set(column_names)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {Path(path).name}: {missing_cols}. "
                    f"Found columns: {column_names}"
                )
        logger.info(f"Total aligned rows to load: {total_aligned_rows:,}")

        # Memory warning only for aligned data (unaligned is streamed)
        estimated_memory_gb = (total_aligned_rows) * 1000 / (1024**3)  # Rough estimate
        logger.info(
            f"Estimated memory usage for aligned data: ~{estimated_memory_gb:.1f}GB "
            f"(unaligned data will be streamed)"
        )

        if estimated_memory_gb > 10:
            logger.warning(
                f"WARNING: Loading aligned data ({total_aligned_rows:,} rows) "
                f"may require ~{estimated_memory_gb:.1f}GB+ of memory."
            )
            response = input("Do you want to continue? (y/n): ").strip().lower()
            if response != "y":
                logger.info("User cancelled operation.")
                sys.exit(0)

        # Load sorted aligned files (only these need to be in memory for searching)
        # Skip sentence_3 to save memory
        logger.info("Loading sorted aligned chunks (skipping sentence_3 column)...")
        columns_to_load = [
            "meta_pmid",
            "meta_language",
            "meta_mesh_ids",
            "meta_mesh_terms",
            "meta_pubdate_year",
            "sentence_1",
            "sentence_2",
        ]
        aligned_chunks = []
        for path in tqdm(sorted_aligned_paths, desc="Loading aligned chunks"):
            chunk = pd.read_parquet(path, columns=columns_to_load)
            logger.info(f"Loaded aligned chunk from {path} with shape {chunk.shape}")
            aligned_chunks.append(chunk)
        df_aligned = pd.concat(aligned_chunks, ignore_index=True)
        logger.info(
            f"Loaded {len(df_aligned)} aligned records, shape: {df_aligned.shape}, columns: {list(df_aligned.columns)}"
        )

        # Initialize ParquetWriter for streaming results
        output_path = self.output_dir / "alignment_results.parquet"
        writer = None
        last_match_idx = 0
        total_processed = 0
        total_matched = 0

        logger.info("Starting iterative alignment process...")

        # Process unaligned files iteratively
        for unaligned_path in sorted_unaligned_paths:
            logger.info(f"Processing unaligned file: {Path(unaligned_path).name}")

            # Read unaligned file in batches (skip sentence_3)
            parquet_file = pq.ParquetFile(unaligned_path)
            columns_to_load_unaligned = ["ID", "sentence_1", "sentence_2"]

            for batch in tqdm(
                parquet_file.iter_batches(
                    batch_size=self.chunk_size, columns=columns_to_load_unaligned
                ),
                desc=f"Processing {Path(unaligned_path).name}",
            ):
                df_batch = batch.to_pandas()
                batch_results = []

                # Process each row in the batch
                for idx, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
                    unaligned_sent1 = row["sentence_1"]
                    unaligned_sent2 = row["sentence_2"]

                    # Skip if sentences are empty
                    if not unaligned_sent2 or unaligned_sent2.strip() == "":
                        batch_results.append(
                            {
                                "id": row["ID"],
                                "meta_pmid": None,
                                "meta_language": None,
                                "meta_mesh_ids": None,
                                "meta_mesh_terms": None,
                                "meta_pubdate_year": None,
                                "match_distance": None,
                                "matched": False,
                            }
                        )
                        continue

                    # Search starting from last match index
                    best_match_idx = None
                    best_distance = float("inf")

                    # Search forward from last match
                    search_start = max(0, last_match_idx)
                    search_end = min(
                        len(df_aligned), last_match_idx + self.search_window
                    )

                    for i in range(search_start, search_end):
                        aligned_sent1 = df_aligned.iloc[i]["sentence_1"]

                        # If first sentences diverge too much alphabetically, we can stop
                        if aligned_sent1 > unaligned_sent1:
                            if i > search_start + 100:
                                break

                        aligned_sent2 = df_aligned.iloc[i]["sentence_2"]

                        if not aligned_sent2 or aligned_sent2.strip() == "":
                            continue

                        # Calculate Levenshtein distance on second sentence
                        dist = levenshtein_distance(unaligned_sent2, aligned_sent2)

                        if dist < best_distance:
                            best_distance = dist
                            best_match_idx = i

                            # If exact match, stop searching
                            if dist == 0:
                                break

                    # Record result
                    if best_match_idx is not None and best_distance <= max_distance:
                        matched_row = df_aligned.iloc[best_match_idx]
                        batch_results.append(
                            {
                                "id": row["ID"],
                                "meta_pmid": matched_row["meta_pmid"],
                                "meta_language": matched_row["meta_language"],
                                "meta_mesh_ids": matched_row["meta_mesh_ids"],
                                "meta_mesh_terms": matched_row["meta_mesh_terms"],
                                "meta_pubdate_year": matched_row["meta_pubdate_year"],
                                "match_distance": best_distance,
                                "matched": True,
                            }
                        )
                        last_match_idx = best_match_idx
                        total_matched += 1
                    else:
                        batch_results.append(
                            {
                                "id": row["ID"],
                                "meta_pmid": None,
                                "meta_language": None,
                                "meta_mesh_ids": None,
                                "meta_mesh_terms": None,
                                "meta_pubdate_year": None,
                                "match_distance": best_distance
                                if best_distance != float("inf")
                                else None,
                                "matched": False,
                            }
                        )

                # Convert batch results to DataFrame and write
                df_batch_results = pd.DataFrame(batch_results)
                total_processed += len(df_batch_results)

                # Convert to PyArrow Table
                table = pa.Table.from_pandas(df_batch_results)

                # Initialize writer on first batch
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                    logger.info(f"Initialized ParquetWriter for {output_path}")

                # Write this batch to the file
                writer.write_table(table)

                # Cleanup
                del df_batch
                del df_batch_results
                del batch_results
                del table
                gc.collect()

        # Close the writer
        if writer is not None:
            writer.close()
            logger.info(f"Closed ParquetWriter. Total rows written: {total_processed}")

        # Print statistics
        logger.info(
            f"Alignment complete: {total_matched}/{total_processed} matched ({total_matched / total_processed * 100:.2f}%)"
        )

        return str(output_path)

    def run_full_pipeline(self, max_distance: int = 10):
        """
        Run the complete alignment pipeline.

        Args:
            max_distance: Maximum Levenshtein distance for matching
        """
        logger.info("Starting full alignment pipeline...")

        # Step 1: Prepare unaligned file
        unaligned_prepared = self.step1_prepare_unaligned_file()

        # Step 2: Prepare alignment file
        aligned_prepared = self.step2_prepare_alignment_file()

        # Step 3: Sort both files
        unaligned_sorted, aligned_sorted = self.step3_sort_by_first_sentence(
            unaligned_prepared, aligned_prepared
        )

        # Step 4: Align using Levenshtein distance
        results_path = self.step4_align_using_levenshtein(
            unaligned_sorted, aligned_sorted, max_distance=max_distance
        )

        logger.info(f"Pipeline complete! Results saved to: {results_path}")
        return results_path


def main():
    """Main function with command-line argument parsing"""

    parser = argparse.ArgumentParser(
        description="Efficiently align unaligned text file with parquet metadata file using sentence sorting and Levenshtein distance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--alignment-file",
        required=True,
        type=str,
        help="Path to parquet file containing metadata (with columns: meta_pmid, meta_language, meta_mesh_ids, meta_mesh_terms, meta_pubdate_year, text)",
    )
    parser.add_argument(
        "--unaligned-file",
        required=True,
        type=str,
        help="Path to .txt file with unaligned texts (one text per line)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Directory to save output files"
    )

    # Optional arguments
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Size of chunks for processing (adjust based on available memory)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=10,
        help="Maximum Levenshtein distance to consider a match",
    )
    parser.add_argument(
        "--search-window",
        type=int,
        default=10000,
        help="Number of records to look ahead from last match position (larger = more thorough but slower)",
    )
    parser.add_argument(
        "--sentence-mode",
        type=str,
        choices=["pysbd", "simple"],
        default="pysbd",
        help="Sentence extraction mode: 'pysbd' (sophisticated) or 'simple' (split by period)",
    )

    args = parser.parse_args()

    # Initialize aligner
    aligner = TextAligner(
        alignment_file_path=args.alignment_file,
        unaligned_file_path=args.unaligned_file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        search_window=args.search_window,
        sentence_mode=args.sentence_mode,
    )

    # Run pipeline
    results_path = aligner.run_full_pipeline(max_distance=args.max_distance)

    # Load and inspect results
    df_results = pd.read_parquet(results_path)
    print(f"\nResults summary:")
    print(f"Total records: {len(df_results)}")
    print(f"Matched: {df_results['matched'].sum()}")
    print(f"Unmatched: {(~df_results['matched']).sum()}")
    print(f"\nMatch distance distribution:")
    print(df_results[df_results["matched"]]["match_distance"].describe())


if __name__ == "__main__":
    main()
