#!/usr/bin/env python3
"""
Utility script to add missing RoBERTa config defaults to saved MultiHead CRF model config.json files.

This script updates existing config.json files to include all necessary RoBERTa parameters
that are required for loading the model with TokenClassificationModelMultiHeadCRF.from_pretrained().

Usage:
    python update_config_defaults.py /path/to/model_directory
    python update_config_defaults.py /path/to/model_directory/config.json
    python update_config_defaults.py /path/to/parent_dir --recursive  # Update all config.json files

Examples:
    # Update a single model
    python update_config_defaults.py /media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/CardioBERTa/multiclass/multihead_maxTL256_batch32_chunk256_epochs10_paper_paragraph_crf/fold_0

    # Update all folds
    python update_config_defaults.py /media/bramiozo/Storage2/DATA/NER/DT4H_results/paper/CardioBERTa/multiclass/multihead_maxTL256_batch32_chunk256_epochs10_paper_paragraph_crf --recursive
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Default RoBERTa parameters needed for model initialization
ROBERTA_DEFAULTS: Dict[str, Any] = {
    # Core model architecture
    "layer_norm_eps": 1e-5,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
    "initializer_range": 0.02,
    "vocab_size": 52000,
    # Token IDs
    "pad_token_id": 1,
    "bos_token_id": 0,
    "eos_token_id": 2,
    # Position embeddings
    "position_embedding_type": "absolute",
    # Model behavior flags
    "use_cache": True,
    "is_decoder": False,
    "add_cross_attention": False,
    "chunk_size_feed_forward": 0,
    "output_hidden_states": False,
    "output_attentions": False,
    "torchscript": False,
    "tie_word_embeddings": True,
    "return_dict": True,
    # Gradient checkpointing
    "gradient_checkpointing": False,
    # Pruning
    "pruned_heads": {},
    # Problem type (for classification)
    "problem_type": None,
    # Embedding layer norm
    "embedding_size": None,
}


def update_config_file(config_path: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Update a config.json file with missing RoBERTa defaults.

    Args:
        config_path: Path to the config.json file
        dry_run: If True, don't write changes, just report what would be done

    Returns:
        Dictionary with 'added' keys and their values
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read existing config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Track what we're adding
    added = {}

    # Add missing defaults
    for key, default_value in ROBERTA_DEFAULTS.items():
        if key not in config or config[key] is None:
            # Special handling: don't override classifier_dropout if it exists with a value
            if (
                key == "classifier_dropout"
                and key in config
                and config[key] is not None
            ):
                continue
            config[key] = default_value
            added[key] = default_value

    if added:
        if not dry_run:
            # Create backup
            backup_path = config_path.with_suffix(".json.bak")
            if not backup_path.exists():
                with open(backup_path, "w", encoding="utf-8") as f:
                    # Re-read original to backup
                    with open(config_path, "r", encoding="utf-8") as orig:
                        f.write(orig.read())

            # Write updated config
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    return added


def find_config_files(base_path: str, recursive: bool = False) -> List[Path]:
    """
    Find config.json files in the given path.

    Args:
        base_path: Base path to search
        recursive: If True, search recursively for all config.json files

    Returns:
        List of paths to config.json files
    """
    base_path = Path(base_path)

    if base_path.is_file() and base_path.name == "config.json":
        return [base_path]

    if base_path.is_dir():
        direct_config = base_path / "config.json"
        if direct_config.exists() and not recursive:
            return [direct_config]

        if recursive:
            return list(base_path.rglob("config.json"))
        elif direct_config.exists():
            return [direct_config]

    return []


def is_multihead_crf_config(config_path: Path) -> bool:
    """Check if a config.json is for a MultiHead CRF model."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return (
            config.get("model_type") == "multihead-crf-tagger"
            or "TokenClassificationModelMultiHeadCRF" in config.get("architectures", [])
            or "entity_types" in config
        )
    except (json.JSONDecodeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update MultiHead CRF model config.json files with missing RoBERTa defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to model directory, config.json file, or parent directory (with --recursive)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively find and update all config.json files",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't make changes, just show what would be done",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Update all config.json files, not just MultiHead CRF ones",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about changes",
    )

    args = parser.parse_args()

    # Find config files
    config_files = find_config_files(args.path, args.recursive)

    if not config_files:
        print(f"No config.json files found in: {args.path}")
        return 1

    print(f"Found {len(config_files)} config.json file(s)")

    updated_count = 0
    skipped_count = 0

    for config_path in config_files:
        # Check if it's a MultiHead CRF config (unless --all is specified)
        if not args.all and not is_multihead_crf_config(config_path):
            if args.verbose:
                print(f"  Skipping (not MultiHead CRF): {config_path}")
            skipped_count += 1
            continue

        try:
            added = update_config_file(config_path, dry_run=args.dry_run)

            if added:
                action = "Would add" if args.dry_run else "Added"
                print(f"\n{action} {len(added)} parameter(s) to: {config_path}")
                if args.verbose:
                    for key, value in added.items():
                        print(f"    {key}: {value}")
                updated_count += 1
            else:
                if args.verbose:
                    print(f"  No changes needed: {config_path}")

        except Exception as e:
            print(f"  Error processing {config_path}: {e}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'}: {updated_count} file(s)")
    if skipped_count:
        print(f"Skipped (not MultiHead CRF): {skipped_count} file(s)")

    if args.dry_run and updated_count > 0:
        print("\nRun without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    exit(main())
