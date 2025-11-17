#!/usr/bin/env python3
"""
Split a large text file into smaller chunks with a specified number of lines.

This script efficiently handles very large files by reading and writing line by line,
avoiding loading the entire file into memory.
"""

import argparse
import os
from pathlib import Path
from typing import Optional


def split_file(
    input_file: str,
    output_dir: Optional[str] = None,
    lines_per_file: int = 5_000_000,
    output_prefix: Optional[str] = None,
) -> None:
    """
    Split a large text file into smaller chunks.

    Args:
        input_file: Path to the input file
        output_dir: Directory where output files will be saved (default: same as input)
        lines_per_file: Number of lines per output file (default: 5,000,000)
        output_prefix: Prefix for output files (default: input filename without extension)
    """
    input_path = Path(input_file)

    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_file}")

    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set output prefix
    if output_prefix is None:
        output_prefix = input_path.stem

    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Lines per file: {lines_per_file:,}")
    print(f"Output prefix: {output_prefix}")
    print("-" * 50)

    file_counter = 1
    line_counter = 0
    total_lines = 0

    # Open first output file
    output_filename = output_dir / f"{output_prefix}_part_{file_counter:03d}.txt"
    output_file = open(output_filename, "w", encoding="utf-8")
    print(f"Writing to: {output_filename}")

    try:
        with open(input_path, "r", encoding="utf-8") as input_f:
            for line in input_f:
                # Write line to current output file
                output_file.write(line)
                line_counter += 1
                total_lines += 1

                # Check if we need to start a new file
                if line_counter >= lines_per_file:
                    output_file.close()
                    print(
                        f"  Completed: {output_filename.name} ({line_counter:,} lines)"
                    )

                    # Start new output file
                    file_counter += 1
                    line_counter = 0
                    output_filename = (
                        output_dir / f"{output_prefix}_part_{file_counter:03d}.txt"
                    )
                    output_file = open(output_filename, "w", encoding="utf-8")
                    print(f"Writing to: {output_filename}")

                # Progress indicator every million lines
                if total_lines % 1_000_000 == 0:
                    print(f"  Processed {total_lines:,} lines...")

        # Close the last output file
        output_file.close()
        if line_counter > 0:
            print(f"  Completed: {output_filename.name} ({line_counter:,} lines)")

    except Exception as e:
        output_file.close()
        raise e

    print("-" * 50)
    print(f"Split complete!")
    print(f"Total lines processed: {total_lines:,}")
    print(f"Number of output files: {file_counter}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a large text file into smaller chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split file into chunks of 5 million lines (default)
  python split_large_file.py input.txt

  # Specify custom chunk size (e.g., 1 million lines)
  python split_large_file.py input.txt --lines 1000000

  # Specify output directory
  python split_large_file.py input.txt --output-dir ./output_chunks

  # Specify custom output prefix
  python split_large_file.py input.txt --prefix my_data
        """,
    )

    parser.add_argument("input_file", help="Path to the input text file to split")

    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for split files (default: same as input file)",
    )

    parser.add_argument(
        "-l",
        "--lines",
        dest="lines_per_file",
        type=int,
        default=5_000_000,
        help="Number of lines per output file (default: 5,000,000)",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        dest="output_prefix",
        default=None,
        help="Prefix for output files (default: input filename without extension)",
    )

    args = parser.parse_args()

    try:
        split_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
            lines_per_file=args.lines_per_file,
            output_prefix=args.output_prefix,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
