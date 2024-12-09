from os import environ
import spacy

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Literal
from collections import defaultdict
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import argparse
import numpy as np
from functools import partial

from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.utils import shuffle

from loader import annotate_corpus_standard, annotate_corpus_centered, tokenize_and_align_labels, align_labels_with_tokens
from trainer import ModelTrainer

import os
from datetime import datetime

# https://huggingface.co/learn/nlp-course/en/chapter7/2


def prepare(Model: str='CLTL/MedRoBERTa.nl',
            lang: str='nl',
            Corpus_tr: str='../../assets/annotations_from_ann_b1.jsonl',
            Corpus_vl: str='../../assets/annotations_from_ann_b2.jsonl',
            annotation_loc: str='../../assets/annotations_from_ann_tokenized.jsonl',
            label2id: Optional[Dict[str, int]]=None,
            id2label: Optional[Dict[int, str]]=None,
            ChunkSize: int=256,
            max_length: int=514,
            ChunkType: Literal['standard', 'centered']='standard'
         ):

    corpus_tr = []
    # read jsonl
    with open(Corpus_tr, 'r', encoding='utf-8') as fr:
        for line in fr:
            corpus_tr.append(json.loads(line))

    corpus_vl = []
    # read jsonl
    with open(Corpus_vl, 'r', encoding='utf-8') as fr:
        for line in fr:
            corpus_vl.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(Model, add_prefix_space=True)
    tokenizer.model_max_length = max_length

    model_config = AutoConfig.from_pretrained(Model)
    max_model_length = model_config.max_position_embeddings  # 514
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)  # 2
    max_allowed_chunk_size = max_model_length - num_special_tokens  # 512


    # Run the transformation
    if ChunkType == 'standard':
        iob_data_b1, unique_tags  = annotate_corpus_standard(corpus_tr, lang, batch_id="train", chunk_size=ChunkSize, max_allowed_chunk_size=max_allowed_chunk_size)
        iob_data_b2, _unique_tags = annotate_corpus_standard(corpus_vl, lang, batch_id="validation", chunk_size=ChunkSize, max_allowed_chunk_size=max_allowed_chunk_size)
    elif ChunkType == 'centered':
        iob_data_b1, unique_tags  = annotate_corpus_centered(corpus_tr, lang, batch_id="train", chunk_size=ChunkSize)
        iob_data_b2, _unique_tags = annotate_corpus_centered(corpus_vl, lang, batch_id="validation", chunk_size=ChunkSize)

    assert(unique_tags == _unique_tags), "Tags are not the same in both batches"

    label2id = {l:int(c) for c,l in enumerate(unique_tags)}
    id2label = {int(c):l for c,l in enumerate(unique_tags)}

    print("Unique tags: ", unique_tags)

    iob_data = iob_data_b1 + iob_data_b2

    partial_tokenize_and_align_labels = partial(tokenize_and_align_labels,
                                                tokenizer=tokenizer,
                                                label2id=label2id)

    iob_data_dataset = Dataset.from_list(iob_data)
    iob_data_dataset_tokenized = iob_data_dataset.map(
        partial_tokenize_and_align_labels,
        batched=True,
    )

    # given a max_length tokens we want to center the context around all spans in the documents and extract them as a seperate documents. Each separate extraction needs to get a separate sub_id.

    max_seq_length = max(len(entry['input_ids']) for entry in iob_data_dataset_tokenized)
    print(f"Maximum sequence length after tokenization: {max_seq_length}")

    annotation_loc = annotation_loc.replace('.jsonl', f'_chunk{ChunkSize}_{ChunkType}.jsonl')
    with open(annotation_loc, 'w', encoding='utf-8') as fw:
        for entry in iob_data_dataset_tokenized:
            entry.update({'label2id': label2id, 'id2label': id2label})
            json.dump(entry, fw)
            fw.write('\n')

    return iob_data_dataset_tokenized, unique_tags


def train(tokenized_data_tr: List[Dict],
          tokenized_data_vl: List[Dict],
          Model: str='CLTL/MedRoBERTa.nl',
          Splits: List[List[str]] | int = 5,
          num_train_epochs: int=3,
          batch_size: int=32,
          output_dir: str="output"):

    label2id = tokenized_data_tr[0]['label2id']
    id2label = tokenized_data_tr[0]['id2label']

    label2id = {str(k):int(v) for k,v in label2id.items()}
    id2label = {int(k):str(v) for k,v in id2label.items()}

    if tokenized_data_vl is None:
        if isinstance(Splits, int):
            splitter = GroupKFold(n_splits=Splits)
            groups = [entry['gid'] for entry in tokenized_data_tr]
            shuffled_data, shuffled_groups = shuffle(tokenized_data_tr, groups, random_state=42)
            SplitList = list(splitter.split(shuffled_data, groups=shuffled_groups))
        else:
            SplitList = Splits

        print(f"Splitting data into {len(SplitList)} folds")
        for k, (train_idx, test_idx) in enumerate(SplitList):
            TrainClass = ModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None,
                                    model=Model, output_dir=f"{output_dir}/fold_{k}")
            print(f"Training on split {k}")
            train_data = [tokenized_data_tr[i] for i in train_idx]
            test_data = [tokenized_data_tr[i] for i in test_idx]
            TrainClass.train(train_data=train_data, eval_data=test_data)
    else:
        SplitList = Splits
        TrainClass = ModelTrainer(label2id=label2id,
                                  id2label=id2label,
                                  tokenizer=None,
                                  model=Model,
                                  output_dir=output_dir,
                                  num_train_epochs=num_train_epochs,
                                  batch_size=batch_size)
        TrainClass.train(train_data=tokenized_data_tr, eval_data=tokenized_data_vl)


if __name__ == "__main__":
    """
        take in .jsonl with:
        {'tags': [{'start': int, 'end': int, 'tag': str}], 'text':str, 'id': str}

        and output .jsonl with tokenized and aligned data
    """

    argparsers = argparse.ArgumentParser()
    argparsers.add_argument('--Model', type=str, required=True)
    argparsers.add_argument('--lang', type=str, required=True, choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'])
    argparsers.add_argument('--mention_class', type=str, required=True)
    argparsers.add_argument('--Corpus_tr', type=str, default=None)
    argparsers.add_argument('--Corpus_vl', type=str, default=None)
    argparsers.add_argument('--annotation_loc', type=str, default=None)
    argparsers.add_argument('--output_dir', type=str, default=None)
    argparsers.add_argument('--parse_annotations', action="store_true", default=False)
    argparsers.add_argument('--train_model', action="store_true", default=False)
    argparsers.add_argument('--chunk_size', type=int, default=256)
    argparsers.add_argument('--max_token_length', type=int, default=514)
    argparsers.add_argument('--num_splits', type=int, default=1)
    argparsers.add_argument('--chunk_type', type=str, default='standard')
    argparsers.add_argument('--epochs', type=int, default=3)
    argparsers.add_argument('--batch_size', type=int, default=64)
    args = argparsers.parse_args()

    tokenized_data = None
    tags = None

    _model = args.Model
    _mention_class = args.mention_class
    _lang = args.lang if args.lang != 'cz' else 'cs'
    _corpus_tr = args.Corpus_tr
    _corpus_vl = args.Corpus_vl
    _annotation_loc = args.annotation_loc
    OutputDir = args.output_dir
    ChunkSize = args.chunk_size
    max_length = args.max_token_length
    ChunkType = args.chunk_type
    parse_annotations = args.parse_annotations
    train_model = args.train_model
    num_labels = args.num_labels
    num_splits = args.num_splits
    epochs = args.epochs
    batch_size = args.batch_size

    model_name = _model.split("/")[-1]

    if _corpus_tr is None:
        _corpus_tr = f'assets/train/{_lang}/{_mention_class}/annotations.jsonl'
    if _corpus_vl is None:
        _corpus_vl = f'assets/validation/{_lang}/{_mention_class}/annotations.jsonl'
    if _annotation_loc is None:
        _annotation_loc = f'assets/train/{_lang}/{_mention_class}/ann_loc.jsonl'
    if OutputDir is None:
        OutputDir = f'output/{_lang}/{_mention_class}/{model_name}'
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    OutputDir = f"{OutputDir}/{dt_string}"

    OutputDir = os.path.abspath(OutputDir)
    if parse_annotations:
        print("Loading and prepping data..")
        tokenized_data, tags = prepare(Model=_model,
                                       lang=_lang,
                                    Corpus_tr=_corpus_tr,
                                    Corpus_vl=_corpus_vl,
                                    annotation_loc=_annotation_loc,
                                    ChunkSize=ChunkSize,
                                    ChunkType=ChunkType,
                                    max_length=max_length)

    if train_model:
        print("Training the model..")
        if tokenized_data is None:
            with open(_annotation_loc, 'r', encoding='utf-8') as fr:
                tokenized_data = [json.loads(line) for line in fr]

        # check if input_ids and labels have the same length, are smaller than max_length and if the labels are within range
        for entry in tokenized_data:
            assert(len(entry['input_ids']) == len(entry['labels'])), f"Input_ids and labels have different lengths for entry {entry['id']}"
            assert(len(entry['input_ids']) <= max_leModel=_model, Splits=num_splits, output_dir=OutputDir, num_train_epochs=epochs, batch_size=batch_size)ngth), f"Input_ids are longer than max_length for entry {entry['id']}"

            # assert that all labels are within range, >=0 and < num_labels
            for label in entry['labels']:
                assert((label >= 0 and label < num_labels) | (label==-100)), f"Label {label}, {type(label)} is not within range for entry {entry['id']}"

            if len(entry['input_ids']) < max_length:
                print(f"{entry['id']} has length {len(entry['input_ids'])} and is smaller than max_length {max_length}")


        #tokenized_data = Dataset.from_list(tokenized_data)
        tokenized_data_tr = [entry for entry in tokenized_data if entry['batch'] == 'train']
        tokenized_data_vl = [entry for entry in tokenized_data if entry['batch'] == 'validation']

        train(tokenized_data_tr,
              tokenized_data_vl,
              Model=_model,
              Splits=num_splits,
              output_dir=OutputDir,
              num_train_epochs=epochs,
              batch_size=batch_size)
