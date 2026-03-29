#!/usr/bin/env python3
"""
Combine inference TSV files into one deduplicated TSV with output columns:
filename, label, start_span, end_span, text

Input TSV files can be mixed-schema:
- minimal: filename, label
- full: filename, label, start_span, end_span, text
Missing optional columns are filled with empty strings.

Usage:
    python scripts/combine_inference.py \
        --results_folder path/to/results \
        --output_file path/to/combined.tsv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Set, Tuple

# Output schema required by your downstream use
OUTPUT_COLUMNS = ["filename", "label", "start_span", "end_span", "text"]
Row = Tuple[str, str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine TSV inference files from a folder into a deduplicated TSV "
            "with columns: filename, label, start_span, end_span, text."
        )
    )
    parser.add_argument(
        "--results_folder",
        type=Path,
        required=True,
        help="Folder containing .tsv inference result files (non-recursive).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to write the combined TSV.",
    )
    return parser.parse_args()


def find_tsv_files(results_folder: Path) -> List[Path]:
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder does not exist: {results_folder}")
    if not results_folder.is_dir():
        raise NotADirectoryError(f"Results folder is not a directory: {results_folder}")

    tsv_files = sorted(
        p
        for p in results_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".tsv"
    )
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in: {results_folder}")
    return tsv_files


def _validate_columns(fieldnames: List[str] | None, path: Path) -> None:
    if not fieldnames:
        raise ValueError(f"{path}: TSV appears to be empty or missing a header row.")

    required = ["filename", "label"]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {missing}. Found columns: {fieldnames}"
        )


def read_rows(tsv_path: Path) -> Set[Row]:
    rows: Set[Row] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        _validate_columns(reader.fieldnames, tsv_path)

        for line_num, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            start_span = (row.get("start_span") or "").strip()
            end_span = (row.get("end_span") or "").strip()
            text = row.get("text")
            text = "" if text is None else str(text)

            # Skip completely empty lines safely
            if not (filename or label or start_span or end_span or text):
                continue

            # Keep strictness for critical columns so bad rows are visible
            if not filename or not label:
                raise ValueError(
                    f"{tsv_path}:{line_num} has empty required values "
                    f"(filename='{filename}', label='{label}')."
                )

            # start_span/end_span/text are optional in input; default to ""
            rows.add((filename, label, start_span, end_span, text))
    return rows


def _sort_key(r: Row):
    filename, label, start_span, end_span, text = r

    def as_int_or_max(v: str):
        try:
            return int(v)
        except ValueError:
            return float("inf")

    return (
        filename,
        as_int_or_max(start_span),
        as_int_or_max(end_span),
        label,
        text,
        start_span,
        end_span,
    )


def combine_inference_rows(results_folder: Path, output_file: Path) -> List[Row]:
    combined: Set[Row] = set()
    output_abs = output_file.expanduser().absolute()

    for tsv_file in find_tsv_files(results_folder):
        # Avoid re-reading the output file if it is in the same folder
        if tsv_file.expanduser().absolute() == output_abs:
            continue
        combined.update(read_rows(tsv_file))
    return sorted(combined, key=_sort_key)


def write_rows(rows: Iterable[Row], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = combine_inference_rows(args.results_folder, args.output_file)
    write_rows(rows, args.output_file)
    print(f"Wrote {len(rows)} unique rows to {args.output_file}")


if __name__ == "__main__":
    main()
