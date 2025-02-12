
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
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import argparse
import numpy as np
from functools import partial


# def annotate_corpus_paragraph, use split_text
def annotate_corpus_paragraph(corpus,
                    batch_id: str="b1",
                    lang: str="nl",
                    chunk_size: int = 256,
                    max_allowed_chunk_size: int = 300,
                    paragraph_boundary: str = "\n\n",
                    IOB: bool=True):
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

        # Initialize tags for each token with empty list
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with labels
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (token_start >= start and token_end <= end) or \
                   (token_start < start and token_end > start) or \
                   (token_start < end and token_end > end):
                    # Token overlaps with entity boundary
                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type
                    token_tags[i].append(tag_label)

        # Split tokens and tags into chunks of max_tokens without splitting entities
        i = 0
        while i < len(tokens):
            ## TODO: make chunker respect paragraph boundaries: paragraph_boundary
            # go to nearest paragraph_boundary < max_allowed_chunk_size

            end_index = min(i + chunk_size, len(tokens))
            # Adjust end_index to avoid splitting entities
            while end_index < len(tokens) and \
                  any(label.startswith('I-') for label in token_tags[end_index]) and \
                  (end_index - i) < max_allowed_chunk_size:
                end_index += 1

                if tokens[i+1] == paragraph_boundary:
                    break

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

    if IOB: 
        tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    else:
        tag_list = ['O'] + [tag for tag in unique_tags]

    return annotated_data, tag_list

# def annotate_corpus_sentence, use pysbd

def annotate_corpus_standard(corpus,
                    batch_id: str="b1",
                    lang: str="nl",
                    chunk_size: int = 256,
                    max_allowed_chunk_size: int = 450,
                    IOB: bool=True):
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

        # Initialize tags for each token with empty list
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with labels
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (token_start >= start and token_end <= end) or \
                   (token_start < start and token_end > start) or \
                   (token_start < end and token_end > end):
                    # Token overlaps with entity boundary
                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type
                    token_tags[i].append(tag_label)

        # Split tokens and tags into chunks of max_tokens without splitting entities
        i = 0
        while i < len(tokens):
            end_index = min(i + chunk_size, len(tokens))
            # Adjust end_index to avoid splitting entities
            while end_index < len(tokens) and \
                  any(label.startswith('I-') for label in token_tags[end_index]) and \
                  (end_index - i) < max_allowed_chunk_size:
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

    if IOB: 
        tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    else:
        tag_list = ['O'] + [tag for tag in unique_tags]

    return annotated_data, tag_list

def annotate_corpus_centered(corpus,
    batch_id: str="b1",
    lang: str="nl",
    chunk_size: int=512,
    IOB: bool=True):

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

        # Initialize tags for each token as an empty list to allow multiple labels
        token_tags = [[] for _ in range(len(doc))]

        # Annotate each token with IOB tags per label
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]
            unique_tags.add(tag_type)

            # Find tokens that fall within the span
            is_first_token = True
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= start:
                    continue  # Token is before the entity
                if token_start >= end:
                    break  # Token is after the entity
                if (token_start >= start and token_end <= end) or \
                   (token_start < start and token_end > start) or \
                   (token_start < end and token_end > end):
                    # Token overlaps with entity boundary
                    
                    if is_first_token and IOB:
                        tag_label = f"B-{tag_type}"
                        is_first_token = False
                    elif IOB:
                        tag_label = f"I-{tag_type}"
                    else:
                        tag_label = tag_type

                    token_tags[i].append(tag_label)

        # Create chunks centered around each span
        for tag in tags:
            start, end, tag_type = tag["start"], tag["end"], tag["tag"]

            # Find the token indices for the span
            span_start_idx = None
            span_end_idx = None
            for i, (token_start, token_end) in enumerate(token_offsets):
                if token_end > start and span_start_idx is None:
                    span_start_idx = i
                if token_start >= end:
                    span_end_idx = i
                    break

            if span_start_idx is None:
                print(f"Warning: Could not find token indices for span in document ID {entry['id']}")
                print(f"Span start: {start}, Span end: {end}")
                continue

            if span_end_idx is None:
                span_end_idx = len(tokens)  # Span goes to the end of the text

            # Adjust left_context and right_context to avoid splitting entities
            left_context = span_start_idx - (chunk_size // 2)
            right_context = span_end_idx + (chunk_size // 2)

            # Ensure contexts are within bounds
            left_context = max(0, left_context)
            right_context = min(len(tokens), right_context)

            # Adjust left_context to avoid starting in the middle of an entity
            while left_context > 0 and any(label.startswith('I-') for label in token_tags[left_context]) and (right_context - left_context) < chunk_size * 2:
                left_context -= 1

            # Adjust right_context to avoid ending in the middle of an entity
            while right_context < len(tokens) and \
                  any(label.startswith('I-') for label in token_tags[right_context - 1]) and \
                  (right_context - left_context) < chunk_size * 2:
                right_context += 1

            # Limit the chunk size to avoid excessively long sequences
            if (right_context - left_context) > chunk_size * 2:
                right_context = left_context + chunk_size * 2

            chunk_tokens = tokens[left_context:right_context]
            chunk_tags = token_tags[left_context:right_context]

            annotated_data.append({
                "gid": entry["id"],
                "id": entry["id"] + f"_span_{start}_{end}",
                "batch": batch_id,
                "tokens": chunk_tokens,
                "tags": chunk_tags,
            })

    if IOB: 
        tag_list = ['O'] + [f'B-{tag},I-{tag}' for tag in unique_tags]
        tag_list = [tag for sublist in tag_list for tag in sublist.split(',')]
    else:
        tag_list = ['O'] + [tag for tag in unique_tags]
    return annotated_data, tag_list

def count_tokens_with_multiple_labels(annotated_data):
    total_tokens = 0
    total_labeled_tokens = 0
    tokens_with_multiple_labels = 0

    for entry in annotated_data:
        token_tags = entry['tags']
        for token_labels in token_tags:
            total_tokens += 1
            if len(token_labels) > 1:
                tokens_with_multiple_labels += 1
                total_labeled_tokens += 1
            elif (len(token_labels)==1) and (token_labels[0] != 'O'):
                total_labeled_tokens += 1

    print(f"Total tokens: {total_tokens}")
    print(f"Total labeled tokens: {total_labeled_tokens}")
    print(f"Tokens with multiple labels: {tokens_with_multiple_labels}")

    if total_tokens > 0:
        percentage_multilabeled = (tokens_with_multiple_labels / total_tokens) * 100
        percentage_lab_multi = (tokens_with_multiple_labels / total_labeled_tokens) * 100
        print(f"Percentage of tokens with multiple labels: {percentage_multilabeled:.2f}%")
        print(f"Percentage of labeled tokens with multiple labels: {percentage_lab_multi:.2f}%")
    else:
        print("No tokens found.")

def align_labels_with_tokens(labels, word_ids, num_labels, id2label):
    new_labels = []
    current_word = None

    label2id = {label: idx for idx, label in id2label.items()}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            # Special token
            new_labels.append([-100] * num_labels)
        elif word_id != current_word:
            # Start of a new word
            current_word = word_id
            if word_id < len(labels):
                label_ids = labels[word_id]
                label_vector = [1 if idx in label_ids else 0 for idx in range(num_labels)]
            else:
                label_vector = [0] * num_labels
            new_labels.append(label_vector)
        else:
            # Same word as previous token, convert B- to I-
            if word_id < len(labels):
                label_ids = labels[word_id]
                label_names = [id2label[idx] for idx in label_ids]
                # Convert B- labels to I- labels
                converted_label_ids = [label2id[label.replace('B-', 'I-')] if label.startswith('B-') else label2id[label] for label in label_names]
                label_vector = [1 if idx in converted_label_ids else 0 for idx in range(num_labels)]
            else:
                label_vector = [0] * num_labels
            new_labels.append(label_vector)
    return new_labels

def tokenize_and_align_labels(docs, tokenizer, label2id, max_length=None):
    # padding is handled during the collation
    #
    tokenized_inputs = tokenizer(
        docs["tokens"],
        is_split_into_words=True,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        padding=False,
        truncation=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True
    )
    id2label = {idx: label for label, idx in label2id.items()}

    all_labels = []
    for token_tags in docs["tags"]:
        # Convert tag names to IDs, defaulting to 'O' if empty
        labels = [ [label2id[tag] for tag in tags] if tags else [label2id['O']] for tags in token_tags ]
        all_labels.append(labels)

    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        num_labels = len(label2id)
        aligned_labels = align_labels_with_tokens(labels, word_ids, num_labels, id2label)
        new_labels.append(aligned_labels)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
