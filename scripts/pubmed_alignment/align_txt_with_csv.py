#!/usr/bin/env python3
"""
Script to align CSV file with abstracts.txt by adding text from corresponding line numbers.

This script reads a CSV file with columns (pmid, year, txt_line) and an abstracts.txt file,
then creates a new CSV with an additional 'text' column containing the text from the
corresponding line number in abstracts.txt.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def load_abstracts(txt_file_path):
    """
    Load abstracts.txt file and return lines as a list.

    Args:
        txt_file_path: Path to the abstracts.txt file

    Returns:
        List of text lines
    """
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Remove newline characters but keep the text
        lines = [line.rstrip('\n') for line in lines]
        print(f"Loaded {len(lines)} lines from {txt_file_path}")
        return lines
    except FileNotFoundError:
        print(f"Error: File {txt_file_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        sys.exit(1)


def align_csv_with_text(csv_path, txt_path, output_path=None, output_format='csv'):
    """
    Align CSV file with abstracts text file by adding text column.

    Args:
        csv_path: Path to the input CSV file
        txt_path: Path to the abstracts.txt file
        output_path: Path for the output file (optional)
        output_format: Output format - 'csv' or 'parquet' (default: 'csv')
    """
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file {csv_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Check required columns
    required_columns = ['pmid', 'year', 'txt_line']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Load abstracts text file
    abstracts_lines = load_abstracts(txt_path)

    # Add text column by mapping line numbers to text
    def get_text_by_line_number(line_num):
        """Get text from abstracts by line number (0-indexed)."""
        try:
            line_idx = int(line_num)
            if 0 <= line_idx < len(abstracts_lines):
                return abstracts_lines[line_idx]
            else:
                return f"[Line {line_idx} out of range]"
        except (ValueError, TypeError):
            return f"[Invalid line number: {line_num}]"

    # Apply mapping to create text column
    df['text'] = df['txt_line'].apply(get_text_by_line_number)

    # Generate output path if not provided
    if output_path is None:
        csv_path_obj = Path(csv_path)
        extension = 'parquet' if output_format == 'parquet' else 'csv'
        output_path = csv_path_obj.parent / f"{csv_path_obj.stem}_with_text.{extension}"

    # Save the aligned data
    if output_format == 'parquet':
        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"\nAlignment complete! (Parquet format)")
    else:
        df.to_csv(output_path, index=False)
        print(f"\nAlignment complete! (CSV format)")

    print(f"Output saved to: {output_path}")

    # Print statistics
    valid_texts = df[~df['text'].str.startswith('[')]['text']
    print(f"\nStatistics:")
    print(f"- Total rows: {len(df)}")
    print(f"- Successfully matched: {len(valid_texts)}")
    print(f"- Failed matches: {len(df) - len(valid_texts)}")

    # Show sample of the result
    print("\nFirst 5 rows of aligned data:")
    print(df[['pmid', 'year', 'txt_line', 'text']].head())

    return df


def main():
    """Main function to handle command line arguments and run the alignment."""
    parser = argparse.ArgumentParser(
        description='Align CSV file with abstracts.txt by adding text from corresponding line numbers'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        help='Path to the input CSV file with columns: pmid, year, txt_line'
    )
    parser.add_argument(
        '--txt_file',
        type=str,
        help='Path to the abstracts.txt file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path for the output file (default: input_with_text.csv or .parquet)'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Output format: csv or parquet (default: csv)'
    )
    parser.add_argument(
        '--encoding',
        type=str,
        default='utf-8',
        help='Text encoding for reading files (default: utf-8)'
    )

    args = parser.parse_args()

    # Run the alignment
    align_csv_with_text(
        csv_path=args.csv_file,
        txt_path=args.txt_file,
        output_path=args.output,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
