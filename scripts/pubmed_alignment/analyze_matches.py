#!/usr/bin/env python3
"""
Analysis script for PubMed alignment match results.

This script analyzes the output from alignment scripts and provides:
- Match statistics and distributions
- Quality metrics
- Duplicate detection
- Comparison between different runs
- Visualization and reporting
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install numpy matplotlib pandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MatchAnalyzer:
    """Analyze alignment match results."""

    def __init__(self, matches_file: Path):
        """
        Initialize analyzer.

        Args:
            matches_file: Path to JSON file with matches
        """
        self.matches_file = matches_file
        self.matches = []
        self.stats = {}

    def load_matches(self) -> int:
        """Load matches from file."""
        logger.info(f"Loading matches from {self.matches_file}")

        if not self.matches_file.exists():
            logger.error(f"File not found: {self.matches_file}")
            return 0

        with open(self.matches_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    match = json.loads(line)
                    self.matches.append(match)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(self.matches)} matches")
        return len(self.matches)

    def compute_statistics(self) -> Dict:
        """Compute match statistics."""
        if not self.matches:
            return {}

        similarities = [m.get("similarity", 0) for m in self.matches]
        match_types = [m.get("match_type", "unknown") for m in self.matches]

        stats = {
            "total_matches": len(self.matches),
            "unique_pubmed_ids": len(set(m.get("pubmed_id") for m in self.matches)),
            "unique_abstracts": len(set(m.get("abstract_line") for m in self.matches)),
            "similarity": {
                "mean": np.mean(similarities),
                "median": np.median(similarities),
                "std": np.std(similarities),
                "min": np.min(similarities),
                "max": np.max(similarities),
                "q25": np.percentile(similarities, 25),
                "q75": np.percentile(similarities, 75),
            },
            "match_types": dict(Counter(match_types)),
        }

        self.stats = stats
        return stats

    def detect_duplicates(self) -> Dict[str, List]:
        """Detect duplicate matches."""
        duplicates = {
            "duplicate_pubmed_ids": defaultdict(list),
            "duplicate_abstracts": defaultdict(list),
        }

        # Group by PubMed ID
        for i, match in enumerate(self.matches):
            pmid = match.get("pubmed_id")
            if pmid:
                duplicates["duplicate_pubmed_ids"][pmid].append(i)

        # Group by abstract line
        for i, match in enumerate(self.matches):
            abs_line = match.get("abstract_line")
            if abs_line is not None:
                duplicates["duplicate_abstracts"][abs_line].append(i)

        # Filter to only duplicates (more than one match)
        duplicate_pmids = {
            k: v for k, v in duplicates["duplicate_pubmed_ids"].items() if len(v) > 1
        }
        duplicate_abstracts = {
            k: v for k, v in duplicates["duplicate_abstracts"].items() if len(v) > 1
        }

        return {
            "duplicate_pubmed_ids": duplicate_pmids,
            "duplicate_abstracts": duplicate_abstracts,
            "num_duplicate_pubmed_ids": len(duplicate_pmids),
            "num_duplicate_abstracts": len(duplicate_abstracts),
        }

    def get_similarity_bins(self, num_bins: int = 10) -> Dict:
        """Get similarity distribution in bins."""
        similarities = [m.get("similarity", 0) for m in self.matches]

        bins = np.linspace(0, 1, num_bins + 1)
        counts, _ = np.histogram(similarities, bins=bins)

        return {
            "bins": bins.tolist(),
            "counts": counts.tolist(),
            "bin_labels": [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(num_bins)],
        }

    def find_low_quality_matches(self, threshold: float = 0.7) -> List[Dict]:
        """Find matches below similarity threshold."""
        low_quality = [m for m in self.matches if m.get("similarity", 1.0) < threshold]
        return low_quality

    def sample_matches(
        self, n: int = 10, sort_by: str = "similarity", ascending: bool = False
    ) -> List[Dict]:
        """Get sample matches."""
        if sort_by in ["similarity"]:
            sorted_matches = sorted(
                self.matches, key=lambda x: x.get(sort_by, 0), reverse=not ascending
            )
        else:
            sorted_matches = self.matches

        return sorted_matches[:n]

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate text report."""
        if not self.stats:
            self.compute_statistics()

        duplicates = self.detect_duplicates()

        lines = []
        lines.append("=" * 80)
        lines.append("MATCH ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Input file: {self.matches_file}")
        lines.append("")

        # Basic statistics
        lines.append("BASIC STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total matches: {self.stats['total_matches']:,}")
        lines.append(f"Unique PubMed IDs: {self.stats['unique_pubmed_ids']:,}")
        lines.append(f"Unique abstracts matched: {self.stats['unique_abstracts']:,}")
        lines.append("")

        # Similarity statistics
        lines.append("SIMILARITY STATISTICS")
        lines.append("-" * 80)
        sim = self.stats["similarity"]
        lines.append(f"Mean similarity: {sim['mean']:.4f}")
        lines.append(f"Median similarity: {sim['median']:.4f}")
        lines.append(f"Std deviation: {sim['std']:.4f}")
        lines.append(f"Min similarity: {sim['min']:.4f}")
        lines.append(f"Max similarity: {sim['max']:.4f}")
        lines.append(f"25th percentile: {sim['q25']:.4f}")
        lines.append(f"75th percentile: {sim['q75']:.4f}")
        lines.append("")

        # Match types
        lines.append("MATCH TYPES")
        lines.append("-" * 80)
        for match_type, count in sorted(
            self.stats["match_types"].items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / self.stats["total_matches"]) * 100
            lines.append(f"{match_type:<20} {count:>8,} ({percentage:>6.2f}%)")
        lines.append("")

        # Duplicates
        lines.append("DUPLICATE DETECTION")
        lines.append("-" * 80)
        lines.append(
            f"PubMed IDs with multiple matches: {duplicates['num_duplicate_pubmed_ids']:,}"
        )
        lines.append(
            f"Abstracts matched multiple times: {duplicates['num_duplicate_abstracts']:,}"
        )
        lines.append("")

        # Quality assessment
        lines.append("QUALITY ASSESSMENT")
        lines.append("-" * 80)
        low_quality = self.find_low_quality_matches(threshold=0.7)
        lines.append(f"Matches below 0.7 similarity: {len(low_quality):,}")

        if len(low_quality) > 0:
            percentage = (len(low_quality) / self.stats["total_matches"]) * 100
            lines.append(f"  ({percentage:.2f}% of total matches)")
        lines.append("")

        # Sample matches
        lines.append("SAMPLE MATCHES (Top 5 by similarity)")
        lines.append("-" * 80)
        top_matches = self.sample_matches(n=5, sort_by="similarity", ascending=False)
        for i, match in enumerate(top_matches, 1):
            lines.append(f"\n{i}. PubMed ID: {match.get('pubmed_id')}")
            lines.append(f"   Similarity: {match.get('similarity', 0):.4f}")
            lines.append(f"   Match type: {match.get('match_type', 'unknown')}")
            lines.append(f"   Abstract (first 100 chars):")
            lines.append(f"   {match.get('abstract_text', '')[:100]}...")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")

        return report

    def plot_similarity_distribution(self, output_file: Optional[Path] = None):
        """Plot similarity distribution."""
        similarities = [m.get("similarity", 0) for m in self.matches]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram
        axes[0, 0].hist(similarities, bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_xlabel("Similarity")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Similarity Distribution")
        axes[0, 0].axvline(
            np.mean(similarities), color="red", linestyle="--", label="Mean"
        )
        axes[0, 0].axvline(
            np.median(similarities), color="green", linestyle="--", label="Median"
        )
        axes[0, 0].legend()

        # Box plot
        axes[0, 1].boxplot(similarities, vert=True)
        axes[0, 1].set_ylabel("Similarity")
        axes[0, 1].set_title("Similarity Box Plot")

        # Cumulative distribution
        sorted_sim = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
        axes[1, 0].plot(sorted_sim, cumulative)
        axes[1, 0].set_xlabel("Similarity")
        axes[1, 0].set_ylabel("Cumulative Probability")
        axes[1, 0].set_title("Cumulative Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Match type distribution
        match_types = [m.get("match_type", "unknown") for m in self.matches]
        type_counts = Counter(match_types)
        axes[1, 1].bar(type_counts.keys(), type_counts.values())
        axes[1, 1].set_xlabel("Match Type")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Match Type Distribution")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

    def export_to_csv(self, output_file: Path):
        """Export matches to CSV."""
        if not self.matches:
            logger.warning("No matches to export")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.matches)

        # Save to CSV
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Exported {len(df)} matches to {output_file}")


def compare_match_files(file1: Path, file2: Path) -> Dict:
    """Compare two match result files."""
    logger.info(f"Comparing {file1.name} vs {file2.name}")

    analyzer1 = MatchAnalyzer(file1)
    analyzer2 = MatchAnalyzer(file2)

    analyzer1.load_matches()
    analyzer2.load_matches()

    # Get PubMed IDs from each
    ids1 = set(m.get("pubmed_id") for m in analyzer1.matches)
    ids2 = set(m.get("pubmed_id") for m in analyzer2.matches)

    common = ids1 & ids2
    only_1 = ids1 - ids2
    only_2 = ids2 - ids1

    comparison = {
        "file1": file1.name,
        "file2": file2.name,
        "matches_1": len(analyzer1.matches),
        "matches_2": len(analyzer2.matches),
        "unique_ids_1": len(ids1),
        "unique_ids_2": len(ids2),
        "common_ids": len(common),
        "only_in_1": len(only_1),
        "only_in_2": len(only_2),
        "jaccard_similarity": len(common) / len(ids1 | ids2)
        if len(ids1 | ids2) > 0
        else 0,
        "overlap_1": len(common) / len(ids1) if len(ids1) > 0 else 0,
        "overlap_2": len(common) / len(ids2) if len(ids2) > 0 else 0,
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PubMed alignment match results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_matches.py matches.json

  # Generate full report with plots
  python analyze_matches.py matches.json --report report.txt --plot plot.png

  # Export to CSV
  python analyze_matches.py matches.json --csv matches.csv

  # Compare two result files
  python analyze_matches.py matches1.json --compare matches2.json
        """,
    )

    parser.add_argument("matches_file", type=Path, help="Path to matches JSON file")

    parser.add_argument("--report", type=Path, help="Save text report to file")

    parser.add_argument("--plot", type=Path, help="Save plots to file (PNG/PDF)")

    parser.add_argument("--csv", type=Path, help="Export matches to CSV")

    parser.add_argument(
        "--compare", type=Path, help="Compare with another matches file"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for quality assessment (default: 0.7)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top matches to show (default: 10)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.matches_file.exists():
        logger.error(f"File not found: {args.matches_file}")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = MatchAnalyzer(args.matches_file)

        # Load matches
        num_matches = analyzer.load_matches()
        if num_matches == 0:
            logger.error("No matches found in file")
            sys.exit(1)

        # Compute statistics
        analyzer.compute_statistics()

        # Generate report
        report = analyzer.generate_report(args.report)
        if not args.report:
            print(report)

        # Generate plots
        if args.plot:
            analyzer.plot_similarity_distribution(args.plot)

        # Export to CSV
        if args.csv:
            analyzer.export_to_csv(args.csv)

        # Compare files
        if args.compare:
            if not args.compare.exists():
                logger.error(f"Comparison file not found: {args.compare}")
            else:
                comparison = compare_match_files(args.matches_file, args.compare)

                print("\n" + "=" * 80)
                print("FILE COMPARISON")
                print("=" * 80)
                print(f"File 1: {comparison['file1']}")
                print(f"  Total matches: {comparison['matches_1']:,}")
                print(f"  Unique PubMed IDs: {comparison['unique_ids_1']:,}")
                print(f"\nFile 2: {comparison['file2']}")
                print(f"  Total matches: {comparison['matches_2']:,}")
                print(f"  Unique PubMed IDs: {comparison['unique_ids_2']:,}")
                print(f"\nOverlap:")
                print(f"  Common PubMed IDs: {comparison['common_ids']:,}")
                print(f"  Only in File 1: {comparison['only_in_1']:,}")
                print(f"  Only in File 2: {comparison['only_in_2']:,}")
                print(f"  Jaccard similarity: {comparison['jaccard_similarity']:.4f}")
                print(f"  Overlap with File 1: {comparison['overlap_1']:.2%}")
                print(f"  Overlap with File 2: {comparison['overlap_2']:.2%}")
                print("=" * 80)

        logger.info("Analysis complete!")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
