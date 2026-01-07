#!/usr/bin/env python3
"""
Parse and analyze PMC filelist CSV files.

After downloading CSV files using list_pmc_files.py, use this script to:
- Parse the CSV files
- Extract metadata about PMC articles
- Filter by criteria
- Generate reports
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_pmc_csv(csv_path: str) -> List[Dict]:
    """
    Parse a PMC filelist CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries with file information
    """
    articles = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                articles.append(row)

        print(f"Parsed {len(articles)} articles from {csv_path}")
        return articles

    except Exception as e:
        print(f"Error parsing {csv_path}: {e}", file=sys.stderr)
        return []


def get_csv_columns(csv_path: str) -> List[str]:
    """
    Get column names from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of column names
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        print(f"Error reading headers: {e}", file=sys.stderr)
        return []


def filter_by_journal(articles: List[Dict], journal_name: str) -> List[Dict]:
    """
    Filter articles by journal name.

    Args:
        articles: List of article dictionaries
        journal_name: Journal name to filter (case-insensitive)

    Returns:
        Filtered list of articles
    """
    filtered = [
        a
        for a in articles
        if "Journal" in a and journal_name.lower() in a["Journal"].lower()
    ]
    return filtered


def filter_by_year(articles: List[Dict], year: str) -> List[Dict]:
    """
    Filter articles by publication year.

    Args:
        articles: List of article dictionaries
        year: Year to filter (e.g., "2023")

    Returns:
        Filtered list of articles
    """
    filtered = []
    for a in articles:
        # Check different possible date fields
        for field in ["Article File", "Article Citation", "Article Title"]:
            if field in a and year in str(a[field]):
                filtered.append(a)
                break
    return filtered


def get_journal_counts(articles: List[Dict]) -> Dict[str, int]:
    """
    Count articles by journal.

    Args:
        articles: List of article dictionaries

    Returns:
        Dictionary mapping journal names to counts
    """
    counts = {}

    for article in articles:
        journal = article.get("Journal", "Unknown")
        counts[journal] = counts.get(journal, 0) + 1

    return counts


def get_file_paths(articles: List[Dict]) -> List[str]:
    """
    Extract file paths from articles.

    Args:
        articles: List of article dictionaries

    Returns:
        List of file paths
    """
    paths = []

    for article in articles:
        # Try different field names that might contain file paths
        for field in ["Article File", "File", "Path", "Filepath"]:
            if field in article and article[field]:
                paths.append(article[field])
                break

    return paths


def get_pmcids(articles: List[Dict]) -> List[str]:
    """
    Extract PMC IDs from articles.

    Args:
        articles: List of article dictionaries

    Returns:
        List of PMC IDs
    """
    pmcids = []

    for article in articles:
        # Try different field names
        for field in ["PMCID", "PMC ID", "AccessionID", "Article Citation"]:
            if field in article and article[field]:
                value = article[field]
                # Extract PMC ID if it's in the format PMCXXXXXXX
                if "PMC" in str(value):
                    import re

                    match = re.search(r"PMC\d+", str(value))
                    if match:
                        pmcids.append(match.group(0))
                        break

    return pmcids


def export_filtered_csv(
    articles: List[Dict], output_path: str, fields: Optional[List[str]] = None
):
    """
    Export filtered articles to a new CSV file.

    Args:
        articles: List of article dictionaries
        output_path: Path for output CSV file
        fields: List of fields to include (None = all fields)
    """
    if not articles:
        print("No articles to export")
        return

    # Determine fields to export
    if fields is None:
        fields = list(articles[0].keys())

    try:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()

            for article in articles:
                writer.writerow(article)

        print(f"Exported {len(articles)} articles to {output_path}")

    except Exception as e:
        print(f"Error exporting CSV: {e}", file=sys.stderr)


def generate_report(articles: List[Dict]) -> str:
    """
    Generate a text report about the articles.

    Args:
        articles: List of article dictionaries

    Returns:
        Report string
    """
    report = []
    report.append("=" * 80)
    report.append("PMC ARTICLES REPORT")
    report.append("=" * 80)
    report.append(f"Total articles: {len(articles)}")
    report.append("")

    # Journal distribution
    journal_counts = get_journal_counts(articles)
    top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    report.append("Top 10 Journals:")
    report.append("-" * 80)
    for journal, count in top_journals:
        journal_name = journal[:70] if len(journal) > 70 else journal
        report.append(f"  {count:6d}  {journal_name}")

    report.append("")

    # Sample articles
    report.append("Sample Articles (first 5):")
    report.append("-" * 80)
    for i, article in enumerate(articles[:5], 1):
        report.append(f"{i}. {article.get('Article File', 'N/A')}")
        if "Journal" in article:
            report.append(f"   Journal: {article['Journal']}")
        report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def example_usage():
    """Example usage of the parsing functions."""

    # Example CSV file (you would download this first using list_pmc_files.py)
    csv_file = "oa_comm_xml.PMC000xxxxxx.baseline.2025-12-18.filelist.csv"

    if not Path(csv_file).exists():
        print(f"CSV file not found: {csv_file}")
        print("\nTo download, use:")
        print("  from list_pmc_files import download_file")
        print(
            f"  download_file('https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/{csv_file}', '{csv_file}')"
        )
        return

    # Parse the CSV
    print(f"Parsing {csv_file}...")
    articles = parse_pmc_csv(csv_file)

    if not articles:
        print("No articles found")
        return

    # Show available columns
    print("\nAvailable columns:")
    for col in articles[0].keys():
        print(f"  - {col}")

    # Generate report
    print("\n")
    report = generate_report(articles)
    print(report)

    # Example: Filter by journal
    print("\nExample: Filtering by journal containing 'Nature'...")
    nature_articles = filter_by_journal(articles, "Nature")
    print(f"Found {len(nature_articles)} articles")

    # Example: Get all PMC IDs
    print("\nExtracting PMC IDs...")
    pmcids = get_pmcids(articles)
    print(f"Found {len(pmcids)} PMC IDs")
    if pmcids:
        print(f"First 5: {pmcids[:5]}")

    # Example: Export filtered results
    if nature_articles:
        output_file = "nature_articles.csv"
        export_filtered_csv(nature_articles, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and analyze PMC filelist CSV files"
    )
    parser.add_argument("csv_file", nargs="?", help="Path to CSV file to parse")
    parser.add_argument("--journal", help="Filter by journal name (case-insensitive)")
    parser.add_argument("--year", help="Filter by year")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--export", help="Export filtered results to CSV file")
    parser.add_argument("--example", action="store_true", help="Run example usage")

    args = parser.parse_args()

    if args.example or not args.csv_file:
        example_usage()
        sys.exit(0)

    # Parse the CSV file
    articles = parse_pmc_csv(args.csv_file)

    if not articles:
        print("No articles found")
        sys.exit(1)

    # Apply filters
    if args.journal:
        articles = filter_by_journal(articles, args.journal)
        print(
            f"Filtered to {len(articles)} articles from journals matching '{args.journal}'"
        )

    if args.year:
        articles = filter_by_year(articles, args.year)
        print(f"Filtered to {len(articles)} articles from year {args.year}")

    # Generate report
    if args.report:
        report = generate_report(articles)
        print(report)

    # Export
    if args.export:
        export_filtered_csv(articles, args.export)

    # If no specific action, just show summary
    if not (args.report or args.export):
        print(f"\nParsed {len(articles)} articles")
        print("\nUse --report to generate a detailed report")
        print("Use --export OUTPUT.csv to save results")
