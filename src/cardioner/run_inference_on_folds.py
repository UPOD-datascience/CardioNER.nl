# process model folds with inference
import json
import os
from typing import List, Literal, Tuple

from cardioner import main

"""
 splits_file:
     {
     "folds": [
        {
            "train_files": [],
            "val_files": [],
        },
        {
            "train_files": [],
            "val_files": [],
        },
        ...
     ],
     "test_files": []
     }
"""


def make_corpora(bulk_file, splits_file):
    corpus_list = []
    with open(bulk_file, "r", encoding="utf-8") as fr:
        for line in fr:
            corpus_list.append(json.loads(line))

    with open(splits_file, "r", encoding="utf-8") as fr:
        split_data = json.load(fr)
        corpus_folds = split_data["folds"]
        corpus_validation_ids = [
            entry.strip(".txt") for entry in split_data["test_files"]
        ]

    corpus_train_id_lists = [
        [entry.strip(".txt") for entry in fold["train_files"]] for fold in corpus_folds
    ]
    corpus_test_id_lists = [
        [entry.strip(".txt") for entry in fold["val_files"]] for fold in corpus_folds
    ]

    # splits: [([train ids],[test ids]), ([train ids],[test ids])...]
    #
    splits = list(zip(corpus_train_id_lists, corpus_test_id_lists))

    corpus_validation_list = [
        entry for entry in corpus_list if entry["id"] in corpus_validation_ids
    ]

    # print overview of counts per fold
    print(100 * "=")
    print("Overview of counts per fold:")
    print(100 * "=")
    for k, fold in enumerate(corpus_folds):
        print(f"Fold {k}:")
        print(f"  Train: {len(fold['train_files'])}")
        print(f"  Validation: {len(fold['val_files'])}")
    print(f"  Test: {len(corpus_validation_list)}")
    print(100 * "=")
    print(100 * "=")

    return corpus_list, corpus_validation_list, splits


def get_model_folders(model_folder: str, folder_prefix="fold_"):
    return [f for f in os.listdir(model_folder) if f.startswith(folder_prefix)]


def process_splits(
    corpora: List[dict],
    model_list: List[str],
    splits: List[Tuple],
    output_dir,
    lang,
    max_word_per_chunk,
    trust_remote_code,
    strategy,
    pipe,
):

    for k, corpus in enumerate(corpora):
        main.inference(
            corpus,
            model_list[k],
            output_dir,
            lang,
            max_word_per_chunk,
            trust_remote_code,
            strategy,
            pipe,
        )
