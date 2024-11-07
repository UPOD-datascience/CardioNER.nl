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
from collections import defaultdict, Sequence
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification

# https://huggingface.co/learn/nlp-course/en/chapter7/2

tokenizer = AutoTokenizer.from_pretrained('CLTL/MedRoBERTa.nl')

# if file locs not given, generate with sklearn
train_ids = ...
test_ids = ...
val_ids = ...


corpus_b1 = []
# read jsonl
with open('../../assets/annotations_from_ann_b1.jsonl', 'r', encoding='utf-8') as fr:
    for line in fr:
        corpus_b1.append(json.loads(line))

corpus_b2 = []
# read jsonl
with open('../../assets/annotations_from_ann_b2.jsonl', 'r', encoding='utf-8') as fr:
    for line in fr:
        corpus_b2.append(json.loads(line))

class DataCollator():
    def __init__(self, labels: List[str]=['DIS', 'PROC', 'SYMP', 'MED']):
        self.labels = labels
        self.label2id = {l:c for c,l in enumerate(labels)}
        self.id2label = {c:l for l,c in self.label2id.items()}

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        #batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])

def annotate_corpus(corpus, batch_id="b1"):
    annotated_data = []

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
            "ner_tags": [d['tag'] for d in token_data],
        })

    return annotated_data

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

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# Run the transformation
iob_data_b1 = annotate_corpus(corpus_b1, batch_id="b1")
iob_data_b2 = annotate_corpus(corpus_b1, batch_id="b2")
iob_data = iob_data_b1 + iob_data_b2

iob_data_dataset = Dataset.from_list(iob_data)
iob_data_dataset_tokenized = iob_data_dataset.map(
    tokenize_and_align_labels,
    batched=True,
)
