"""
Incoming format is :
[
{"tags": [{"start": xx, "end":xx, "tag": "DISEASE"},...],
 "id": xxx,
 "text": xxx}

]
"""
from ast import parse
from os import truncate
import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Literal
from collections import defaultdict
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import argparse
import numpy as np
from functools import partial

from sklearn.model_selection import train_test_split, KFold

import evaluate
metric = evaluate.load("seqeval")

# https://huggingface.co/learn/nlp-course/en/chapter7/2


class ModelTrainer():
    def __init__(self,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 tokenizer=None,
                 model: str='CLTL/MedRoBERTa.nl',
                 batch_size: int=16,
                 max_length: int=512,
                 learning_rate: float=5e-5,
                 weight_decay: float=0.01,
                 num_train_epochs: int=3,
    ):
        self.label2id = label2id
        self.id2label = id2label

        self.train_kwargs = {
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'weight_decay': weight_decay,
            'eval_strategy':'epoch',
            'save_strategy': 'epoch'
        }

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model,
                add_prefix_space=True, max_length=max_length, padding="max_length")
        else:
            self.tokenizer = tokenizer

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(model, id2label=self.id2label, label2id=self.label2id)

        self.args = TrainingArguments(
            output_dir="../results",
            **self.train_kwargs
        )

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[labels[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def train(self, train_data: List[Dict], eval_data: List[Dict], output_dir: str="../results"):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        metrics = trainer.evaluate()
        try:
            trainer.save_metrics(output_dir, metrics=metrics)
            trainer.save_model(output_dir)
        except:
            print("Failed to save model and metrics")
        return metrics


def annotate_corpus_standard(corpus,
                    batch_id: str="b1", 
                    chunk_size: int = 256):
    annotated_data = []

    unique_tags = set()

    for entry in corpus:
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with "O" (outside)
        token_tags = ["O"] * len(doc)

        # Annotate each token with IOB tags
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if token_start >= start and token_end <= end:
                    # Token is inside the entity
                    if token_tags[i] == "O":
                        token_tags[i] = f"B-{tag_type}"
                    else:
                        token_tags[i] = f"I-{tag_type}"
                elif (token_start < start and token_end > start) or (token_start < end and token_end > end):
                    # Token overlaps with entity boundary
                    token_tags[i] = f"I-{tag_type}"

        # Split tokens and tags into chunks of max_tokens without splitting entities
        i = 0
        while i < len(tokens):
            end_index = min(i + chunk_size, len(tokens))
            # Adjust end_index to avoid splitting entities
            while end_index < len(tokens) and token_tags[end_index].startswith('I-'):
                end_index += 1
            chunk_tokens = tokens[i:end_index]
            chunk_tags = token_tags[i:end_index]

            annotated_data.append({
                "id": entry["id"] + f"_{i//chunk_size}",  # Modify ID to reflect chunk
                "batch": batch_id,
                "tokens": chunk_tokens,
                "tags": chunk_tags,
            })

            i = end_index  # Move to the next chunk

    tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
    tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    return annotated_data, tag_list

def annotate_corpus_centered(corpus, batch_id="b1", chunk_size=512):
    annotated_data = []
    unique_tags = set()

    for entry in corpus:
        text = entry["text"]
        tags = entry["tags"]

        # Tokenize the text using spaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # Initialize tags for each token with "O" (outside)
        token_tags = ["O"] * len(doc)

        # Annotate each token with IOB tags
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if token_start >= start and token_end <= end:
                    # Token is inside the entity
                    if token_tags[i] == "O":
                        token_tags[i] = f"B-{tag_type}"
                    else:
                        token_tags[i] = f"I-{tag_type}"
                elif (token_start < start and token_end > start) or (token_start < end and token_end > end):
                    # Token overlaps with entity boundary
                    token_tags[i] = f"I-{tag_type}"

        # Create chunks centered around each span
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]

            # Find the token indices for the span
            span_start_idx = None
            span_end_idx = None
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end >= start and span_start_idx is None:
                    span_start_idx = i
                if token_start >= end:
                    span_end_idx = i
                    break

            if span_start_idx is None:
                print(f"Warning: Could not find token indices for span in document ID {entry['id']}")
                print(f"Span start: {start}, Span end: {end}")
                #print(f"Token offsets: {token_offsets}")
                continue

            if span_end_idx is None:
                span_end_idx = len(tokens)  # Span goes to the end of the text

            # Center the chunk around the span
            left_context = max(0, span_start_idx - (chunk_size // 2))
            right_context = min(len(tokens), span_start_idx + (chunk_size // 2))

            # Adjust if the span is near the start or end of the document
            if right_context - left_context < chunk_size:
                if left_context == 0:
                    right_context = min(len(tokens), left_context + chunk_size)
                elif right_context == len(tokens):
                    left_context = max(0, right_context - chunk_size)

            chunk_tokens = tokens[left_context:right_context]
            chunk_tags = token_tags[left_context:right_context]

            annotated_data.append({
                "id": entry["id"] + f"_span_{start}_{end}",
                "batch": batch_id,
                "tokens": chunk_tokens,
                "tags": chunk_tags,
            })

    tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
    tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    return annotated_data, tag_list


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(docs, tokenizer, label2id: Optional[Dict[str, int]]=None):
    '''
    Tokenizes and aligns labels with tokens.
    '''
    tokenized_inputs = tokenizer(
        docs["tokens"], is_split_into_words=True
    )
    if label2id is not None:
        # Corrected this line to handle lists of lists
        all_labels = [[label2id[tag] for tag in tags] for tags in docs["tags"]]
    else:
        all_labels = docs["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def prepare(Model: str='CLTL/MedRoBERTa.nl',
         Corpus_b1: str='../../assets/annotations_from_ann_b1.jsonl',
         Corpus_b2: str='../../assets/annotations_from_ann_b2.jsonl',
         annotation_loc: str='../../assets/annotations_from_ann_tokenized.jsonl',
         label2id: Optional[Dict[str, int]]=None,
         id2label: Optional[Dict[int, str]]=None,
         ChunkSize: int=256,
         ChunkType: Literal['standard', 'centered']='standard'
         ):

    corpus_b1 = []
    # read jsonl
    with open(Corpus_b1, 'r', encoding='utf-8') as fr:
        for line in fr:
            corpus_b1.append(json.loads(line))

    corpus_b2 = []
    # read jsonl
    with open(Corpus_b2, 'r', encoding='utf-8') as fr:
        for line in fr:
            corpus_b2.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(Model, add_prefix_space=True)

    # Run the transformation
    if ChunkType == 'standard':
        iob_data_b1, unique_tags  = annotate_corpus_standard(corpus_b1, batch_id="b1", chunk_size=ChunkSize)
        iob_data_b2, _unique_tags = annotate_corpus_standard(corpus_b2, batch_id="b2", chunk_size=ChunkSize)
    elif ChunkType == 'centered':
        iob_data_b1, unique_tags  = annotate_corpus_centered(corpus_b1, batch_id="b1", chunk_size=ChunkSize)
        iob_data_b2, _unique_tags = annotate_corpus_centered(corpus_b2, batch_id="b2", chunk_size=ChunkSize)

    assert(unique_tags == _unique_tags), "Tags are not the same in both batches"

    label2id = {l:c for c,l in enumerate(unique_tags)}
    id2label = {c:l for c,l in enumerate(unique_tags)}

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


def train(tokenized_data: List[Dict],
          Model: str='CLTL/MedRoBERTa.nl',
          Splits: List[List[str]] | int = 5):

    label2id = tokenized_data[0]['label2id']
    id2label = tokenized_data[0]['id2label']

    if isinstance(Splits, int):
        splitter = KFold(n_splits=Splits, shuffle=True, random_state=42)
        # get the splits
        SplitList = list(splitter.split(tokenized_data))
    else:
        SplitList = Splits

    TrainClass = ModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None, model=Model)

    for train_idx, test_idx in SplitList:
        train_data = [tokenized_data[i] for i in train_idx]
        test_data = [tokenized_data[i] for i in test_idx]
        TrainClass.train(train_data=train_data, eval_data=test_data)


if __name__ == "__main__":
    """
        take in .jsonl with:
        {'tags': [{'start': int, 'end': int, 'tag': str}], 'text':str, 'id': str}

        and output .jsonl with tokenized and aligned data
    """

    argparsers = argparse.ArgumentParser()
    argparsers.add_argument('--Model', type=str, default='CLTL/MedRoBERTa.nl')
    argparsers.add_argument('--Corpus_b1', type=str, default='../../assets/annotations_from_ann_b1.jsonl')
    argparsers.add_argument('--Corpus_b2', type=str, default='../../assets/annotations_from_ann_b2.jsonl')
    argparsers.add_argument('--annotation_loc', type=str, default='../../assets/annotations_from_ann_tokenized.jsonl')
    argparsers.add_argument('--parse_annotations', action="store_true", default=False)
    argparsers.add_argument('--train_model', action="store_true", default=False)
    argparsers.add_argument('--chunk_size', type=int, default=256)
    argparsers.add_argument('--chunk_type', type=str, default='standard')
    args = argparsers.parse_args()


    tokenized_data = None
    tags = None

    _model = args.Model
    _corpus_b1 = args.Corpus_b1
    _corpus_b2 = args.Corpus_b2
    _annotation_loc = args.annotation_loc
    ChunkSize = args.chunk_size
    ChunkType = args.chunk_type
    parse_annotations = args.parse_annotations
    train_model = args.train_model

    if parse_annotations:
        print("Loading and prepping data..")
        tokenized_data, tags = prepare(Model=_model,
                                    Corpus_b1=_corpus_b1,
                                    Corpus_b2=_corpus_b2,
                                    annotation_loc=_annotation_loc,
                                    ChunkSize=ChunkSize,
                                    ChunkType=ChunkType)

    if train_model:
        print("Training the model..")
        if tokenized_data is None:
            with open(_annotation_loc, 'r', encoding='utf-8') as fr:
                tokenized_data = [json.loads(line) for line in fr]
        #tokenized_data = Dataset.from_list(tokenized_data)
        train(tokenized_data, Model=_model, Splits=10)
