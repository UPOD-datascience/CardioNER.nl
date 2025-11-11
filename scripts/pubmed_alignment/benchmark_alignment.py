#!/usr/bin/env python3
"""
Benchmarking script for comparing different PubMed alignment strategies.

This script helps you:
1. Test different strategies on sample data
2. Measure performance (speed, memory, accuracy)
3. Compare results and get recommendations
4. Validate matches manually
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import tempfile
from collections import defaultdict

try:
    import psutil
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install psutil numpy tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics during execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update(self):
        """Update peak memory."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        elapsed_time = time.time() - self.start_time
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = self.peak_memory - self.start_memory

        return {
            "elapsed_time": elapsed_time,
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_used_mb": memory_used,
        }


class BenchmarkRunner:
    """Run benchmark tests for different alignment strategies."""

    def __init__(
        self,
        abstracts_file: Path,
        dataset_path: str,
        text_column: str,
        id_column: str,
        mesh_id_column: str,
        sample_size: int = 1000,
    ):
        self.abstracts_file = abstracts_file
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.id_column = id_column
        self.mesh_id_column = mesh_id_column
        self.sample_size = sample_size
        self.results = {}

    def create_sample_data(self) -> Tuple[Path, Path]:
        """Create sample abstracts file for benchmarking."""
        logger.info(f"Creating sample data ({self.sample_size} abstracts)...")

        sample_abstracts = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        )

        with open(self.abstracts_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.sample_size:
                    break
                sample_abstracts.write(line)

        sample_abstracts.close()
        return Path(sample_abstracts.name)

    def run_original_strategy(
        self, sample_file: Path, output_file: Path
    ) -> Dict[str, float]:
        """Run original align_pubmed.py strategy."""
        logger.info("Testing original strategy...")

        monitor = PerformanceMonitor()
        monitor.start()

        try:
            cmd = [
                sys.executable,
                "align_pubmed.py",
                "--abstracts_file",
                str(sample_file),
                "--dataset_path",
                self.dataset_path,
                "--text-column",
                self.text_column,
                "--id-column",
                self.id_column,
                "--mesh-ids-column",
                self.mesh_id_column,
                "--output",
                str(output_file),
                "--debug",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )  # 1 hour timeout

            if result.returncode != 0:
                logger.warning(f"Original strategy failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("Original strategy timed out (1 hour)")
            return None
        except Exception as e:
            logger.warning(f"Original strategy error: {e}")
            return None

        metrics = monitor.stop()
        metrics["matches"] = self._count_matches(output_file)
        return metrics

    def run_optimized_strategy(
        self, strategy: str, sample_file: Path, output_file: Path, threshold: float
    ) -> Dict[str, float]:
        """Run optimized strategy."""
        logger.info(f"Testing optimized strategy: {strategy}")

        monitor = PerformanceMonitor()
        monitor.start()

        try:
            cmd = [
                sys.executable,
                "align_pubmed_optimized.py",
                "--abstracts",
                str(sample_file),
                "--dataset",
                self.dataset_path,
                "--text-column",
                self.text_column,
                "--id-column",
                self.id_column,
                "--mesh-ids-column",
                self.mesh_id_column,
                "--strategy",
                strategy,
                "--threshold",
                str(threshold),
                "--output",
                str(output_file),
                "--debug",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )  # 1 hour timeout

            if result.returncode != 0:
                logger.warning(f"Strategy {strategy} failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Strategy {strategy} timed out")
            return None
        except Exception as e:
            logger.warning(f"Strategy {strategy} error: {e}")
            return None

        metrics = monitor.stop()
        metrics["matches"] = self._count_matches(output_file)
        return metrics

    def run_embedding_strategy(
        self, sample_file: Path, output_file: Path, model: str, threshold: float
    ) -> Dict[str, float]:
        """Run embedding-based strategy."""
        logger.info(f"Testing embedding strategy with model: {model}")

        monitor = PerformanceMonitor()
        monitor.start()

        try:
            cmd = [
                sys.executable,
                "align_pubmed_embeddings.py",
                "--abstracts",
                str(sample_file),
                "--dataset",
                self.dataset_path,
                "--text-column",
                self.text_column,
                "--id-column",
                self.id_column,
                "--mesh-ids-column",
                self.mesh_id_column,
                "--model",
                model,
                "--threshold",
                str(threshold),
                "--output",
                str(output_file),
                "--debug",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )  # 1 hour timeout

            if result.returncode != 0:
                logger.warning(f"Embedding strategy failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("Embedding strategy timed out")
            return None
        except Exception as e:
            logger.warning(f"Embedding strategy error: {e}")
            return None

        metrics = monitor.stop()
        metrics["matches"] = self._count_matches(output_file)
        return metrics

    def _count_matches(self, output_file: Path) -> int:
        """Count matches in output file."""
        if not output_file.exists():
            return 0

        count = 0
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.warning(f"Error counting matches: {e}")

        return count

    def compare_results(
        self, file1: Path, file2: Path, sample_size: int = 10
    ) -> Dict[str, float]:
        """Compare results from two output files."""
        matches1 = self._load_matches(file1)
        matches2 = self._load_matches(file2)

        # Find common and unique matches
        ids1 = {m["pubmed_id"] for m in matches1}
        ids2 = {m["pubmed_id"] for m in matches2}

        common = ids1 & ids2
        only_1 = ids1 - ids2
        only_2 = ids2 - ids1

        # Calculate similarity metrics
        jaccard = len(common) / len(ids1 | ids2) if len(ids1 | ids2) > 0 else 0
        overlap_1 = len(common) / len(ids1) if len(ids1) > 0 else 0
        overlap_2 = len(common) / len(ids2) if len(ids2) > 0 else 0

        return {
            "total_matches_1": len(matches1),
            "total_matches_2": len(matches2),
            "common_matches": len(common),
            "only_in_1": len(only_1),
            "only_in_2": len(only_2),
            "jaccard_similarity": jaccard,
            "overlap_rate_1": overlap_1,
            "overlap_rate_2": overlap_2,
        }

    def _load_matches(self, file_path: Path) -> List[Dict]:
        """Load matches from file."""
        matches = []
        if not file_path.exists():
            return matches

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        matches.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error loading matches from {file_path}: {e}")

        return matches

    def run_full_benchmark(
        self, strategies: List[str], threshold: float = 0.7
    ) -> Dict[str, Dict]:
        """Run full benchmark comparing all strategies."""
        sample_file = self.create_sample_data()
        results = {}

        try:
            # Test each strategy
            for strategy in strategies:
                output_file = Path(f"benchmark_{strategy}.json")

                if strategy == "original":
                    metrics = self.run_original_strategy(sample_file, output_file)
                elif strategy.startswith("embedding_"):
                    model = strategy.replace("embedding_", "")
                    metrics = self.run_embedding_strategy(
                        sample_file, output_file, model, threshold
                    )
                else:
                    metrics = self.run_optimized_strategy(
                        strategy, sample_file, output_file, threshold
                    )

                if metrics:
                    results[strategy] = {
                        "metrics": metrics,
                        "output_file": output_file,
                    }

                # Clean up output file
                if output_file.exists() and strategy != "original":
                    # Keep original for comparison
                    pass

        finally:
            # Clean up sample file
            if sample_file.exists():
                sample_file.unlink()

        return results

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("BENCHMARK RESULTS")
        report.append("=" * 80)
        report.append(f"Sample size: {self.sample_size} abstracts")
        report.append(f"Dataset: {self.dataset_path}")
        report.append("")

        # Performance comparison table
        report.append("Performance Metrics:")
        report.append("-" * 80)
        report.append(
            f"{'Strategy':<20} {'Time (s)':<12} {'Memory (MB)':<15} {'Matches':<10}"
        )
        report.append("-" * 80)

        for strategy, data in sorted(results.items()):
            metrics = data["metrics"]
            report.append(
                f"{strategy:<20} "
                f"{metrics['elapsed_time']:<12.2f} "
                f"{metrics['memory_used_mb']:<15.2f} "
                f"{metrics['matches']:<10}"
            )

        report.append("")

        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 80)

        # Find fastest
        fastest = min(results.items(), key=lambda x: x[1]["metrics"]["elapsed_time"])
        report.append(
            f"Fastest: {fastest[0]} ({fastest[1]['metrics']['elapsed_time']:.2f}s)"
        )

        # Find most matches
        most_matches = max(results.items(), key=lambda x: x[1]["metrics"]["matches"])
        report.append(
            f"Most matches: {most_matches[0]} ({most_matches[1]['metrics']['matches']} matches)"
        )

        # Find most memory efficient
        most_efficient = min(
            results.items(), key=lambda x: x[1]["metrics"]["memory_used_mb"]
        )
        report.append(
            f"Most memory efficient: {most_efficient[0]} ({most_efficient[1]['metrics']['memory_used_mb']:.2f} MB)"
        )

        # Estimate full dataset time
        report.append("")
        report.append("Estimated time for full dataset (10M records):")
        report.append("-" * 80)
        for strategy, data in sorted(results.items()):
            metrics = data["metrics"]
            time_per_record = metrics["elapsed_time"] / self.sample_size
            estimated_time = time_per_record * 10_000_000  # 10M records
            hours = estimated_time / 3600
            report.append(f"{strategy:<20} {hours:<10.2f} hours")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different PubMed alignment strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with default strategies
  python benchmark_alignment.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids

  # Test specific strategies
  python benchmark_alignment.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids \\
      --strategies exact_hash lsh hybrid

  # Large sample for more accurate results
  python benchmark_alignment.py \\
      --abstracts abstracts.txt \\
      --dataset my_dataset \\
      --text-column abstract \\
      --id-column pmid \\
      --mesh-ids-column mesh_ids \\
      --sample-size 10000
        """,
    )

    parser.add_argument(
        "--abstracts", type=Path, required=True, help="Path to abstracts file"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset path"
    )

    parser.add_argument(
        "--text-column", type=str, required=True, help="Text column name"
    )

    parser.add_argument(
        "--id-column", type=str, required=True, help="PubMed ID column name"
    )

    parser.add_argument(
        "--mesh-ids-column", type=str, required=True, help="MeSH IDs column name"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["exact_hash", "lsh", "tfidf_blocking", "hybrid"],
        help="Strategies to benchmark (default: exact_hash lsh tfidf_blocking hybrid)",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of abstracts to sample (default: 1000)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="benchmark_report.txt",
        help="Output report file (default: benchmark_report.txt)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.abstracts.exists():
        logger.error(f"Abstracts file not found: {args.abstracts}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("STARTING BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Abstracts file: {args.abstracts}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Strategies: {', '.join(args.strategies)}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            abstracts_file=args.abstracts,
            dataset_path=args.dataset,
            text_column=args.text_column,
            id_column=args.id_column,
            mesh_id_column=args.mesh_ids_column,
            sample_size=args.sample_size,
        )

        # Run benchmark
        results = runner.run_full_benchmark(args.strategies, args.threshold)

        # Generate and save report
        report = runner.generate_report(results)
        print("\n" + report)

        with open(args.output, "w") as f:
            f.write(report)

        logger.info(f"\nReport saved to: {args.output}")

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
