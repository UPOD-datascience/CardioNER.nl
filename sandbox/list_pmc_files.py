#!/usr/bin/env python3
"""
List and download files from NCBI PMC Open Access FTP server via HTTPS.
Uses only standard library (no external dependencies).
Note: pandas functions require pandas to be installed.
"""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin
from urllib.request import urlopen


class LinkParser(HTMLParser):
    """Parse HTML and extract links from <a> tags."""

    def __init__(self):
        super().__init__()
        self.links = []
        self.current_link = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, value in attrs:
                if attr == "href":
                    self.current_link = value
                    break

    def handle_data(self, data):
        if self.current_link:
            self.links.append((self.current_link, data.strip()))
            self.current_link = None


def list_pmc_files(
    url: str = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/",
    pattern: Optional[str] = None,
    filename_pattern: Optional[str] = None,
) -> List[str]:
    """
    List files from NCBI PMC directory via HTTPS.

    Args:
        url: URL of the directory
        pattern: Filter files by extension or content (e.g., '.csv', '.tar.gz')
        filename_pattern: Additional filter - filename must contain this string

    Returns:
        List of filenames
    """
    print(f"Fetching directory listing from {url}")

    try:
        with urlopen(url, timeout=30) as response:
            html_content = response.read().decode("utf-8")

        # Parse HTML to extract links
        parser = LinkParser()
        parser.feed(html_content)

        files = []
        for href, text in parser.links:
            # Skip parent directory and absolute URLs
            if href.startswith("/") or href.startswith("http") or href == "../":
                continue

            # Filter by pattern if provided
            if pattern and pattern not in href:
                continue

            # Filter by filename pattern if provided
            if filename_pattern and filename_pattern not in href:
                continue

            files.append(href)

        print(f"Found {len(files)} files")
        return files

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []


def list_pmc_files_detailed(
    url: str = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/",
    pattern: Optional[str] = None,
    filename_pattern: Optional[str] = None,
) -> List[Dict]:
    """
    List files with detailed information (size, date, etc.).

    Args:
        url: URL of the directory
        pattern: Filter files by extension or content (e.g., '.csv', '.tar.gz')
        filename_pattern: Additional filter - filename must contain this string

    Returns:
        List of dicts with file info
    """
    print(f"Fetching directory listing from {url}")

    try:
        with urlopen(url, timeout=30) as response:
            html_content = response.read().decode("utf-8")

        # Parse the pre-formatted directory listing
        # Format: <a href="filename">filename</a>  date  size
        files_info = []

        # Use regex to extract file info from Apache-style directory listing
        # Pattern matches: href="filename" followed by date and size
        pattern_re = r'<a href="([^"]+)">.*?</a>\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+(\d+\.?\d*)([KMG]?)\s*$'

        for line in html_content.split("\n"):
            match = re.search(pattern_re, line)
            if not match:
                continue

            href = match.group(1)
            date = match.group(2)
            size_num = float(match.group(3))
            size_unit = match.group(4)

            # Skip parent directory and absolute URLs
            if href.startswith("/") or href.startswith("http") or href == "../":
                continue

            # Filter by pattern if provided
            if pattern and pattern not in href:
                continue

            # Filter by filename pattern if provided
            if filename_pattern and filename_pattern not in href:
                continue

            # Convert size to bytes
            size = 0
            if size_unit == "K":
                size = int(size_num * 1024)
            elif size_unit == "M":
                size = int(size_num * 1024 * 1024)
            elif size_unit == "G":
                size = int(size_num * 1024 * 1024 * 1024)
            else:
                size = int(size_num)

            size_mb = size / (1024 * 1024)
            size_gb = size / (1024 * 1024 * 1024)

            files_info.append(
                {
                    "name": href,
                    "url": urljoin(url, href),
                    "size": size,
                    "size_mb": round(size_mb, 2),
                    "size_gb": round(size_gb, 2),
                    "date": date,
                }
            )

        print(f"Found {len(files_info)} files")
        return files_info

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL.

    Args:
        url: URL to download
        output_path: Local path to save file
        chunk_size: Size of chunks to download

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {url} to {output_path}...")

        with urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            with open(output_path, "wb") as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"Progress: {percent:.1f}% ({downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB)",
                            end="\r",
                        )

        print(f"\nDownload complete: {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading file: {e}", file=sys.stderr)
        return False


def load_and_concatenate_csvs(
    csv_files: Union[List[str], List[Dict]],
    url_base: Optional[str] = None,
    download_dir: Optional[str] = None,
    pattern: Optional[str] = None,
    filename_pattern: Optional[str] = None,
    cache_to_disk: bool = True,
    verbose: bool = True,
):
    """
    Load and concatenate multiple PMC CSV files into a single pandas DataFrame.

    Args:
        csv_files: List of filenames (strings) or file info dicts from list_pmc_files_detailed()
        url_base: Base URL for downloading (if csv_files are filenames, not dicts with 'url')
        download_dir: Directory to download files to if cache_to_disk=True (default: current directory)
        pattern: Filter files by extension or content (e.g., '.csv', '2025-12-18')
        filename_pattern: Additional filter - filename must contain this string
        cache_to_disk: If True, cache downloaded files to disk. If False, load directly to memory.
        verbose: Print progress messages

    Returns:
        pandas.DataFrame with concatenated data

    Raises:
        ImportError: If pandas is not installed

    Example:
        # Load to memory only (no caching)
        df = load_and_concatenate_csvs(files, cache_to_disk=False)

        # Cache to disk (default behavior)
        df = load_and_concatenate_csvs(files, download_dir='./data')
    """
    try:
        from io import BytesIO

        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_and_concatenate_csvs(). "
            "Install it with: pip install pandas"
        )

    # Setup download directory if caching
    if cache_to_disk:
        if download_dir is None:
            download_dir = "."
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)

    # Filter files by pattern if provided
    if pattern or filename_pattern:
        filtered_files = []
        for f in csv_files:
            filename = f["name"] if isinstance(f, dict) else f

            if pattern and pattern not in filename:
                continue
            if filename_pattern and filename_pattern not in filename:
                continue

            filtered_files.append(f)
        csv_files = filtered_files

        if verbose:
            print(f"Filtered to {len(csv_files)} files")

    if not csv_files:
        if verbose:
            print("No CSV files to load")
        return pd.DataFrame()

    dataframes = []

    for i, file_info in enumerate(csv_files, 1):
        # Extract filename and URL
        if isinstance(file_info, dict):
            filename = file_info["name"]
            file_url = file_info.get("url")
        else:
            filename = file_info
            if url_base:
                file_url = urljoin(url_base, filename)
            else:
                file_url = None

        # Check if we should use cached file
        if cache_to_disk:
            local_path = download_path / filename
            use_cached = local_path.exists()
        else:
            local_path = None
            use_cached = False

        # Load from cache or download
        if use_cached:
            if verbose:
                print(f"[{i}/{len(csv_files)}] Loading {filename} (from cache)...")
            try:
                df = pd.read_csv(local_path)
            except Exception as e:
                if verbose:
                    print(f"Error loading cached file {filename}: {e}")
                continue

        else:
            # Download to memory or disk
            if file_url is None:
                if verbose:
                    print(f"Skipping {filename}: no URL provided")
                continue

            if verbose:
                if cache_to_disk:
                    print(
                        f"[{i}/{len(csv_files)}] Downloading {filename} to {download_dir}..."
                    )
                else:
                    print(f"[{i}/{len(csv_files)}] Downloading {filename} to memory...")

            try:
                with urlopen(file_url, timeout=30) as response:
                    content = response.read()

                if cache_to_disk:
                    # Write to disk
                    with open(local_path, "wb") as f:
                        f.write(content)
                    # Load from disk
                    df = pd.read_csv(local_path)
                else:
                    # Load directly from memory
                    df = pd.read_csv(BytesIO(content))

            except Exception as e:
                if verbose:
                    print(f"Error downloading/loading {filename}: {e}")
                continue

        # Add source filename as a column
        df["_source_file"] = filename
        dataframes.append(df)

        if verbose:
            print(f"  Loaded {len(df)} rows")

    if not dataframes:
        if verbose:
            print("No dataframes loaded successfully")
        return pd.DataFrame()

    # Concatenate all dataframes
    if verbose:
        print(f"\nConcatenating {len(dataframes)} dataframes...")

    combined_df = pd.concat(dataframes, ignore_index=True)

    if verbose:
        print(
            f"Combined DataFrame: {len(combined_df)} rows, {len(combined_df.columns)} columns"
        )
        print(f"Columns: {list(combined_df.columns)}")

    return combined_df


if __name__ == "__main__":
    # Example 1: List all CSV files
    print("=" * 80)
    print("LISTING ALL CSV FILES")
    print("=" * 80)
    csv_files = list_pmc_files(pattern=".csv")

    for i, f in enumerate(csv_files[:20], 1):  # Show first 20
        print(f"{i:3}. {f}")

    if len(csv_files) > 20:
        print(f"... and {len(csv_files) - 20} more files")

    print("\n")

    # Example 2: List only latest .tar.gz files using filename_pattern
    print("=" * 80)
    print("LISTING LATEST .tar.gz FILES (2025-12-18)")
    print("=" * 80)
    tar_files = list_pmc_files(pattern=".tar.gz", filename_pattern="2025-12-18")

    for i, f in enumerate(tar_files[:10], 1):  # Show first 10
        print(f"{i:3}. {f}")

    if len(tar_files) > 10:
        print(f"... and {len(tar_files) - 10} more files")

    print("\n")

    # Example 3: Get detailed info for CSV files
    print("=" * 80)
    print("DETAILED INFORMATION FOR CSV FILES")
    print("=" * 80)

    detailed_csv = list_pmc_files_detailed(pattern=".csv")

    # Sort by name
    detailed_csv.sort(key=lambda x: x["name"])

    print(f"{'File':<60} {'Size (MB)':<12} {'Date':<20}")
    print("-" * 92)

    for file_info in detailed_csv[:20]:  # Show first 20
        print(
            f"{file_info['name']:<60} {file_info['size_mb']:<12.2f} {file_info['date']:<20}"
        )

    if len(detailed_csv) > 20:
        print(f"... and {len(detailed_csv) - 20} more files")

    print("\n")

    # Example 4: Get detailed info for .tar.gz files
    print("=" * 80)
    print("DETAILED INFORMATION FOR .tar.gz FILES (sorted by size)")
    print("=" * 80)

    detailed_tar = list_pmc_files_detailed(pattern=".tar.gz")

    # Sort by size (largest first)
    detailed_tar.sort(key=lambda x: x["size"], reverse=True)

    print(f"{'File':<60} {'Size (GB)':<12} {'Date':<20}")
    print("-" * 92)

    for file_info in detailed_tar[:15]:  # Show first 15
        print(
            f"{file_info['name']:<60} {file_info['size_gb']:<12.2f} {file_info['date']:<20}"
        )

    if len(detailed_tar) > 15:
        print(f"... and {len(detailed_tar) - 15} more files")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total CSV files: {len(detailed_csv)}")
    print(f"Total .tar.gz files: {len(detailed_tar)}")
    total_size_gb = sum(f["size_gb"] for f in detailed_tar)
    print(f"Total .tar.gz size: {total_size_gb:.2f} GB")

    # Example of how to download a file
    print("\n" + "=" * 80)
    print("EXAMPLE: Download a specific CSV file")
    print("=" * 80)
    if detailed_csv:
        example_file = detailed_csv[0]
        print(f"To download: {example_file['name']}")
        print(f"URL: {example_file['url']}")
        print(f"\nIn Python:")
        print(f"  from list_pmc_files import download_file")
        print(f"  download_file('{example_file['url']}', '{example_file['name']}')")

    # Example 7: Load and concatenate CSVs with pandas
    print("\n" + "=" * 80)
    print("EXAMPLE: Load and concatenate CSV files with pandas")
    print("=" * 80)
    print("To load and combine CSV files into a pandas DataFrame:")
    print(
        "\n  from list_pmc_files import list_pmc_files_detailed, load_and_concatenate_csvs"
    )
    print("  ")
    print("  # Get latest CSV files for specific PMC ranges")
    print(
        '  files = list_pmc_files_detailed(pattern=".csv", filename_pattern="2025-12-18")'
    )
    print(
        "  pmc_range = [f for f in files if 'PMC000' in f['name'] or 'PMC001' in f['name']]"
    )
    print("  ")
    print("  # Load and concatenate")
    print("  df = load_and_concatenate_csvs(pmc_range[:3], download_dir='./data')")
    print("  print(df.head())")
    print("\nNote: Requires pandas to be installed (pip install pandas)")
