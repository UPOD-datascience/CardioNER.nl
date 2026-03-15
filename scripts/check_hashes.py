# a script to compare the hashes for .txt files in subfolders
# there are four subfolders per folder dis, med, proc, symp
# they all contain a "txt" folder containing the txt files.
#
#

#!/usr/bin/env python3
"""Compare the hashes for .txt files across annotation subfolders.

Each annotation folder (dis, med, proc, symp) contains a "txt" subfolder
with the source text files. Since all categories share the same source
documents, the hashes should be identical across subfolders. This script
verifies that assumption and reports any mismatches.
"""

import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

LABEL_FOLDERS = ["dis", "med", "proc", "symp"]


def compute_file_hash(
    filepath: Path, algorithm: str = "sha256", strip: bool = False
) -> str:
    """Compute the hash of a file using the specified algorithm."""
    h = hashlib.new(algorithm)
    if strip:
        content = filepath.read_bytes().strip()
        h.update(content)
    else:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return h.hexdigest()


def read_text(filepath: Path) -> str:
    """Read a text file with replacement for invalid characters."""
    return filepath.read_text(encoding="utf-8", errors="replace")


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    return Levenshtein.distance(a, b)


def collect_hashes(
    folder: Path, subfolders: list[str], algorithm: str, strip: bool = False
) -> dict[str, dict[str, dict[str, object]]]:
    """Collect hashes for all .txt files in each subfolder.

    Returns a nested dict: {subfolder: {filename: {"hash": str, "path": Path}}}.
    """
    result: dict[str, dict[str, dict[str, object]]] = {}
    for sub in subfolders:
        txt_dir = folder / sub / "txt"
        if not txt_dir.is_dir():
            print(f"Warning: directory not found: {txt_dir}", file=sys.stderr)
            result[sub] = {}
            continue
        result[sub] = {
            p.name.replace(sub, ""): {
                "hash": compute_file_hash(p, algorithm, strip=strip),
                "path": p,
            }
            for p in sorted(txt_dir.glob("*.txt"))
        }
    return result


def compare_hashes(
    hashes: dict[str, dict[str, dict[str, object]]],
    verbose: bool = False,
    csv_rows: list[dict[str, object]] | None = None,
) -> int:
    """Compare hashes across subfolders and report mismatches.

    Returns the number of files with issues.
    """
    subfolders = list(hashes.keys())
    if len(subfolders) < 2:
        print("Need at least two subfolders to compare.", file=sys.stderr)
        return 0

    # Gather all unique filenames
    all_files: set[str] = set()
    for sub_hashes in hashes.values():
        all_files.update(sub_hashes.keys())

    mismatches = 0
    missing = 0

    for filename in sorted(all_files):
        present_in = {
            sub: hashes[sub][filename]["hash"]
            for sub in subfolders
            if filename in hashes[sub]
        }
        absent_from = [sub for sub in subfolders if filename not in hashes[sub]]

        # Report missing files
        if absent_from:
            missing += 1
            print(f"MISSING  {filename}")
            print(f"  present in : {', '.join(sorted(present_in.keys()))}")
            print(f"  absent from: {', '.join(sorted(absent_from))}")
            continue

        # Compare all hashes to each other
        unique_hashes = set(present_in.values())
        if len(unique_hashes) == 1:
            if verbose:
                print(f"OK       {filename}  ({list(unique_hashes)[0][:12]}...)")
            if csv_rows is not None:
                row: dict[str, object] = {"filename": filename}
                for left, right in itertools.combinations(subfolders, 2):
                    row[f"distance_{left}_vs_{right}"] = 0
                csv_rows.append(row)
        else:
            mismatches += 1
            print(f"MISMATCH {filename}")
            for sub in subfolders:
                print(f"  {sub:6s} : {present_in[sub]}")

            texts = {
                sub: read_text(hashes[sub][filename]["path"]) for sub in subfolders
            }
            distances: dict[tuple[str, str], int] = {}
            for left, right in itertools.combinations(subfolders, 2):
                distance = levenshtein(texts[left], texts[right])
                distances[(left, right)] = distance
                print(f"  levenshtein {left} vs {right}: {distance}")

            if csv_rows is not None:
                row: dict[str, object] = {"filename": filename}
                for left, right in itertools.combinations(subfolders, 2):
                    row[f"distance_{left}_vs_{right}"] = distances[(left, right)]
                csv_rows.append(row)

    return mismatches + missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare file hashes for .txt files across annotation subfolders "
            "(dis, med, proc, symp). Verifies that the source text files are "
            "identical across all categories."
        ),
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help=(
            "Path to the parent folder that contains the annotation "
            "subfolders (e.g. 'data/b1/1_validated_without_sugs/es')."
        ),
    )
    parser.add_argument(
        "-s",
        "--subfolders",
        nargs="+",
        default=LABEL_FOLDERS,
        metavar="NAME",
        help=(f"Annotation subfolders to compare. Default: {' '.join(LABEL_FOLDERS)}"),
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        default="sha256",
        choices=hashlib.algorithms_available,
        help="Hash algorithm to use (default: sha256).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print filenames that match across all subfolders.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip leading and trailing whitespace from file contents before hashing.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Write mismatched file Levenshtein distances to a CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing .txt hashes in: {folder}")
    print(f"Subfolders : {', '.join(args.subfolders)}")
    print(f"Algorithm  : {args.algorithm}")
    print(f"Strip      : {args.strip}")
    print()

    hashes = collect_hashes(folder, args.subfolders, args.algorithm, strip=args.strip)

    # Print per-subfolder file counts
    for sub in args.subfolders:
        count = len(hashes.get(sub, {}))
        print(f"  {sub:6s} : {count} file(s)")
    print()

    csv_rows: list[dict[str, object]] | None = [] if args.csv else None
    issues = compare_hashes(hashes, verbose=args.verbose, csv_rows=csv_rows)

    if args.csv is not None and csv_rows is not None:
        fieldnames = ["filename"] + [
            f"distance_{left}_vs_{right}"
            for left, right in itertools.combinations(args.subfolders, 2)
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    if issues == 0:
        print("\nAll files match across subfolders.")
    else:
        print(f"\n{issues} file(s) with issues found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
