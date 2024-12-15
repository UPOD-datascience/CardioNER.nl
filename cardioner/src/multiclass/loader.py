
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


def annotate_corpus_standard(corpus,
                    batch_id: str="b1",
                    lang: str="nl",
                    chunk_size: int = 256,
                    max_allowed_chunk_size: int = 450):
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

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
            while end_index < len(tokens) and token_tags[end_index].startswith('I-') and (end_index - i) < max_allowed_chunk_size:
                end_index += 1
            # Ensure the chunk does not exceed max_allowed_chunk_size
            if (end_index - i) > max_allowed_chunk_size:
                end_index = i + max_allowed_chunk_size
            chunk_tokens = tokens[i:end_index]
            chunk_tags = token_tags[i:end_index]

            annotated_data.append({
                "gid": entry["id"],
                "id": entry["id"] + f"_{i//chunk_size}",  # Modify ID to reflect chunk
                "batch": batch_id,
                "tokens": chunk_tokens,
                "tags": chunk_tags,
            })

            i = end_index

    tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
    tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    return annotated_data, tag_list

def annotate_corpus_centered(corpus, lang:str='nl', batch_id="b1", chunk_size=512):
    annotated_data = []
    unique_tags = set()

    nlp = spacy.blank(lang)

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
                "gid": entry["id"],
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
    max_label_idx = len(labels) - 1  # Maximum valid index for labels
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            # Ensure word_id does not exceed labels length
            if word_id > max_label_idx:
                label = -100  # Assign -100 if word_id is beyond labels
            else:
                label = labels[word_id]
            new_labels.append(label)
        else:
            # Same word as previous token
            if word_id > max_label_idx:
                label = -100
            else:
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(docs,
                              tokenizer, label2id: Optional[Dict[str, int]]=None,
                              max_length: Optional[int]=None):
    '''
    Tokenizes and aligns labels with tokens.
    '''
    tokenized_inputs = tokenizer(
        docs["tokens"],
        is_split_into_words=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        padding='max_length',      # Pad sequences to max_length
        truncation=True,           # Truncate sequences longer than max_length
        return_offsets_mapping=True
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
