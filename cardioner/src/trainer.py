"""
Incoming format is :
[
{"tags": [{"start": xx, "end":xx, "tag": "DISEASE"},...],
 "id": xxx,
 "text": xxx}

]
"""
import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
from pydantic import BaseModel
from typing import List, Dict, Optional
from collections import defaultdict
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
import argparse
import numpy as np
from functools import partial

import evaluate
metric = evaluate.load("seqeval")

# https://huggingface.co/learn/nlp-course/en/chapter7/2


class ModelTrainer():
    def __init__(self, labels: List[str]=['O', 'B-DIS', 'I-DIS', 'B-PROC', 'I-PROC', 'B-SYMP', 'I-SYMP', 'B-MED', 'I-MED'], 
                 tokenizer=None, 
                 model: str='CLTL/MedRoBERTa.nl',):
        self.labels = labels
        self.label2id = {l:c for c,l in enumerate(labels)}
        self.id2label = {c:l for l,c in self.label2id.items()}

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        else:
            self.tokenizer = tokenizer

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(model, id2label=self.id2label, label2id=self.label2id)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.labels[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }


def annotate_corpus(corpus, batch_id="b1"):
    annotated_data = []

    unique_tags = set()
    for entry in corpus:
        text = entry["text"]
        tags = entry["tags"]

        # Initialize tags for each token with "O" (outside)
        doc = nlp(text)
        token_tags = ["O"] * len(doc)

        # Annotate each span with IOB tags
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            token_start = None
            token_end = None
            
            unique_tags.add(tag_type)

            for token in doc:
                if token.idx >= start and token.idx < end:
                    if token_start is None:
                        token_start = token.i
                        token_tags[token.i] = f"B-{tag_type}"
                    else:
                        token_tags[token.i] = f"I-{tag_type}"
                elif token.idx >= end:
                    break

        # Compile each token, its tag, and token ID
        token_data = [
            {"token": token.text, "tag": token_tags[i], "token_id": token.i}
            for i, token in enumerate(doc)
        ]

        annotated_data.append({
            "id": entry["id"],
            "batch": batch_id,
            "tokens": [d['token'] for d in token_data],
            "tags": [d['tag'] for d in token_data],
        })
    

    tag_list = ",".join(['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]).split(",")
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
        docs["tokens"], truncation=True, is_split_into_words=True
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
         Output: str='../../assets/annotations_from_ann_tokenized.jsonl',
         label2id: Optional[Dict[str, int]]=None, 
         id2label: Optional[Dict[int, str]]=None
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
    iob_data_b1, unique_tags  = annotate_corpus(corpus_b1, batch_id="b1")
    iob_data_b2, _unique_tags = annotate_corpus(corpus_b2, batch_id="b2")
    
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

    with open(Output, 'w', encoding='utf-8') as fw:
        for entry in iob_data_dataset_tokenized:
            json.dump(entry, fw)
            fw.write('\n')

    return iob_data_dataset_tokenized, unique_tags


def train(tokenized_data, unique_tags, Model: str='CLTL/MedRoBERTa.nl'):
    print("Not implemented yet")
    pass


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
    argparsers.add_argument('--Output', type=str, default='../../assets/annotations_from_ann_tokenized.jsonl')

    kwargs = vars(argparsers.parse_args())

    print("Loading and prepping data..")
    tokenized_data, tags = prepare(**kwargs)

    train(tokenized_data, tags, Model='CLTL/MedRoBERTa.nl')
