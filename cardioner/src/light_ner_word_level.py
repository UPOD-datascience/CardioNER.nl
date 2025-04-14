import lightning as L

import itertools
import re
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import spacy

from transformers import PreTrainedModel, AutoTokenizer, AutoConfig, AutoModel
from transformers import RobertaModel, BertModel, PreTrainedTokenizer
from transformers import RobertaForTokenClassification, BertForTokenClassification
from transformers import XLMRobertaForTokenClassification
from tokenizers import Encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, F1Score, Precision, Recall, Accuracy
from torch import Tensor
from torch.nn import ModuleDict
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import os
from functools import partial
import gc
from evaluate import load
from torchmetrics import Metric
import lightning as L
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import classification_report

torch.cuda.empty_cache()
gc.collect()

import argparse


def rename_tag(tag):
    if tag == "PROCEDIMIENTO":
        return "PROCEDURE"
    elif tag == "SINTOMA":
        return "SYMPTOM"
    elif tag == "FARMACO":
        return "MEDICATION"
    elif tag == "ENFERMEDAD":
        return "DISEASE"
    return tag


class CardioCCC(Dataset):
    LABEL_FOLDERS = ["dis", "med", "symp", "proc"]

    def __init__(
        self,
        root_path: str,
        split: str,
        tag2label: dict,
        lang: str = "it",
        encoding: str = "UTF-8",
        with_suggestion: bool = False,
        iob_tags: bool = False,
    ):

        val_str = "2_validated_w_sugs" if with_suggestion else "1_validated_without_sugs"
        self.root_path = Path(root_path)
        self.split_file_names = json.load((self.root_path / "splits.json").open())[lang][split]["symp"]
        self.lang = lang
        self.iob_tags = iob_tags
        batches = ["b1", "b2"] if lang != "ro" else ["b1"]
        self.annotations = []
        self.nlp = spacy.blank(lang if lang != "cz" else "cs")
        for batch in batches:
            lang_path = self.root_path / batch / val_str / lang
            raw_annotations = []
            for label_folder in self.LABEL_FOLDERS:
                ann_path = lang_path / label_folder / "tsv"
                raw_annotations.append(pd.read_csv(next(ann_path.glob("*.tsv")), sep="\t", na_filter=False))
            raw_annotations = pd.concat(raw_annotations, axis=0)
            if lang == "es":
                raw_annotations["tag"] = raw_annotations["tag"].apply(rename_tag)

            for group in raw_annotations.groupby("name"):
                if group[0] not in self.split_file_names:
                    continue
                file_name = group[0] + ".txt"
                text = (lang_path / "dis/txt" / file_name).read_text(encoding=encoding)
                labels = group[1].loc[:, ["tag", "start_span", "end_span", "text"]].to_dict(orient="records")
                tokens, labels = tokenize_and_align_labels(self.nlp, text, labels, self.iob_tags, tag2label)
                self.annotations.append({"tokens": tokens, "labels": labels})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]


def tokenize_and_align_labels(nlp, text, labels, iob_tags, tag2label: dict):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]
    token_tags = {tag: ["O"] * len(doc) for tag in tag2label.keys()}
    D_REMAP = {"ENFERMEDAD": "DISEASE", "SINTOMA": "SYMPTOM", "PROCEDIMIENTO": "PROCEDURE", "FARMACO": "MEDICATION"}
    for label in labels:
        tag = D_REMAP.get(label["tag"], label["tag"])
        start, end, tag_type = int(label["start_span"]), int(label["end_span"]), tag
        is_first_token = True
        for i, (token_start, token_end) in enumerate(token_offsets):
            if token_end <= start:
                continue  # Token is before the entity
            if token_start >= end:
                break  # Token is after the entity
            if (
                (token_start >= start and token_end <= end)
                or (token_start < start and token_end > start)
                or (token_start < end and token_end > end)
            ):
                # Token overlaps with entity boundary
                if is_first_token and iob_tags:
                    tag_label = f"B-{tag_type}"
                    is_first_token = False
                else:
                    tag_label = f"I-{tag_type}"
                token_tags[tag_type][i] = tag_label

    return tokens, token_tags


def flatten_token_list(l):
    if isinstance(l, list):
        if l and isinstance(l[0], str):
            return [l]
    return [item for sublist in l for item in flatten_token_list(sublist)]


def split_tokens(tokens: list[str], tokenizer: PreTrainedTokenizer, max_seq_len: int):
    paragraphs_breaks = [0] + [i + 1 for i in range(len(tokens)) if re.match(".*\n\n+.*", tokens[i])] + [len(tokens)]
    paragraphs = [tokens[paragraphs_breaks[i] : paragraphs_breaks[i + 1]] for i in range(len(paragraphs_breaks) - 1)]
    for p_idx, para_tokens in enumerate(paragraphs):
        ids = tokenizer.encode(para_tokens, add_special_tokens=False, is_split_into_words=True)
        if len(ids) > max_seq_len:
            line_breaks = (
                [0]
                + [i + 1 for i in range(len(para_tokens)) if re.sub("\n\n+", "", para_tokens[i]).count("\n") == 1]
                + [len(para_tokens)]
            )
            lines = [para_tokens[line_breaks[i] : line_breaks[i + 1]] for i in range(len(line_breaks) - 1)]
            for l_idx, line_tokens in enumerate(lines):
                ids = tokenizer.encode(line_tokens, add_special_tokens=False, is_split_into_words=True)
                if len(ids) > max_seq_len:
                    sentence_breaks = (
                        [0]
                        + [i + 1 for i in range(len(line_tokens)) if re.match(r"[\.!\?]", line_tokens[i])]
                        + [len(line_tokens)]
                    )
                    sentences = [
                        line_tokens[sentence_breaks[i] : sentence_breaks[i + 1]] for i in range(len(sentence_breaks) - 1)
                    ]
                    for s_idx, sentence_tokens in enumerate(sentences):
                        ids = tokenizer.encode(sentence_tokens, add_special_tokens=False, is_split_into_words=True)
                        if len(ids) > max_seq_len:
                            word_breaks = [i for i in range(len(sentence_tokens) + 1)]
                            words = [sentence_tokens[word_breaks[i] : word_breaks[i + 1]] for i in range(len(word_breaks) - 1)]
                            sentences[s_idx] = words
                    lines[l_idx] = sentences
            paragraphs[p_idx] = lines
    splits = flatten_token_list(paragraphs)
    return splits


def align_labels_to_text(encoding: Encoding, token_tags: dict[str, list[str]], tag2label: dict, iob_tags: bool):
    offset = 3 if iob_tags else 2
    num_labels = len(tag2label.keys()) * offset
    labels = torch.zeros((encoding["input_ids"].shape[1], num_labels))
    for i in range(len(token_tags[list(token_tags.keys())[0]])):
        tok_span = encoding.word_to_tokens(i)
        if tok_span is None:
            for tag in token_tags.keys():
                assert token_tags[tag][i] == "O"
        else:
            start, end = tok_span
            for tag in token_tags.keys():
                if token_tags[tag][i] == "O":
                    labels[start:end, tag2label[tag] * offset] = 1
                elif token_tags[tag][i].startswith("I"):
                    labels[start:end, tag2label[tag] * offset + 1] = 1
                elif iob_tags and token_tags[tag][i].startswith("B"):
                    labels[start:end, tag2label[tag] * offset + 2] = 1
    return labels


def get_tokens_indices(word_to_token_list: list[int], start_idx: int, end_idx: int):
    token_idx_list = []
    for i in range(start_idx, end_idx):
        tok_span = word_to_token_list[i]
        if tok_span is not None:
            start, end = tok_span
            for i in range(start, end):
                token_idx_list.append(i)
    # token_idx_list = [k for k, _ in itertools.groupby(token_idx_list)]
    return token_idx_list


def merge_splits_into_chunks(
    tokens: list[str],
    splits: list[str],
    token_tags: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    tag2label: dict,
    iob_tags: bool,
):

    encoding = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
    tokens_ids = encoding.input_ids[0]
    word_to_token_list = [encoding.word_to_tokens(i) for i in range(len(tokens))]

    label_ids = align_labels_to_text(encoding, token_tags, tag2label, iob_tags)
    assert len(tokens_ids) == len(label_ids)
    # Merge splits into chunks without exceeding max_seq_len
    start_chunk_idx, end_chunk_idx = 0, 0
    chunks = {"tokens": [], "input_ids": [], "label_ids": [], "word_to_token": [], "labels_words": []}
    for i in range(len(splits) + 1):
        # TODO: optimize this
        if i < len(splits):
            # Compute the current chunk length after adding the next tokenized split
            sentence = splits[i]
            token_idx_list = get_tokens_indices(word_to_token_list, start_chunk_idx, end_chunk_idx + len(sentence))
            chunk_ids = tokens_ids[token_idx_list]
        if i == len(splits) or len(chunk_ids) > max_seq_len:
            # add previous splits as a chunk if current chunk exceeds max_seq_len or if the splits are finished
            token_idx_list = get_tokens_indices(word_to_token_list, start_chunk_idx, end_chunk_idx)
            chunk_ids = tokens_ids[token_idx_list]
            chunk_labels_ids = label_ids[token_idx_list]
            chunk_labels_words = {
                tag: [token_tags[tag][i] for i in range(start_chunk_idx, end_chunk_idx)] for tag in token_tags.keys()
            }
            chunk_word_to_token = []
            for i in range(start_chunk_idx, end_chunk_idx):
                tok_span = word_to_token_list[i]
                if tok_span is not None:
                    start, end = tok_span
                    chunk_word_to_token.append([j - token_idx_list[0] for j in range(start, end)])
                else:
                    chunk_word_to_token.append(None)

            chunks["tokens"].append(tokens[start_chunk_idx:end_chunk_idx])
            chunks["input_ids"].append(chunk_ids)
            chunks["label_ids"].append(chunk_labels_ids)
            chunks["word_to_token"].append(chunk_word_to_token)
            chunks["labels_words"].append(chunk_labels_words)
            start_chunk_idx = end_chunk_idx
        end_chunk_idx += len(sentence)
    return chunks


class ChunkedCardioCCC(Dataset):
    def __init__(
        self,
        dataset: CardioCCC,
        tokenizer: PreTrainedTokenizer,
        language: str,
        tag2label: dict,
        iter_by_chunk: bool = False,
        model_max_len: int = 512,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.language = language
        self.chunked_data = []
        self.iter_by_chunk = iter_by_chunk
        self.tag2label = tag2label
        for item in dataset:
            tokens, labels = item["tokens"], item["labels"]
            splits = split_tokens(tokens, tokenizer, model_max_len)
            chunks = merge_splits_into_chunks(tokens, splits, labels, tokenizer, model_max_len, tag2label, self.dataset.iob_tags)
            if iter_by_chunk:
                for i in range(len(chunks["tokens"])):
                    self.chunked_data.append(
                        {
                            "tokens": chunks["tokens"][i],
                            "input_ids": chunks["input_ids"][i],
                            "label_ids": chunks["label_ids"][i],
                            "word_to_token": chunks["word_to_token"][i],
                            "labels_words": chunks["labels_words"][i],
                        }
                    )
            else:
                self.chunked_data.append(chunks)

    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self, idx):
        return self.chunked_data[idx]


def collate_fn_chunked_bert(batch: list[dict], padding_value: int):
    input_ids = [chunk["input_ids"] for chunk in batch]
    labels = [chunk["label_ids"] for chunk in batch]
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    word_to_token = [chunk["word_to_token"] for chunk in batch]
    labels_words = [chunk["labels_words"] for chunk in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "word_to_token": word_to_token,
        "labels_words": labels_words,
    }


def collate_fn_chunked_bert_test(chunks: list[dict], padding_value: int):
    chunks = chunks[0]
    num_chunks = len(chunks["input_ids"])
    input_ids = [chunks["input_ids"][i] for i in range(num_chunks)]
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=padding_value)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_to_token": chunks["word_to_token"],
        "labels_words": chunks["labels_words"],
    }


class NEREval(Metric):
    def __init__(self, iob_tags: bool, tag2label: dict):
        super().__init__()
        self.iob_tags = iob_tags
        self.tag2label = tag2label
        self.preds = []
        self.labels_words = []
        self.seqeval = load("seqeval")
        self.metrics = {}

    def update(self, preds: list[Tensor], labels_words: list[dict]) -> None:
        self.preds.append(preds)
        self.labels_words.append(labels_words)

    def compute(self):
        preds, labels_words = self.preds, self.labels_words
        offset = 3 if self.iob_tags else 2
        results = {}
        predictions, references = {tag: [] for tag in self.tag2label.keys()}, {tag: [] for tag in self.tag2label.keys()}

        for i in range(len(preds)):
            for tag in self.tag2label.keys():
                tag_idx = self.tag2label[tag]
                tag_preds = preds[i][:, tag_idx * offset : (tag_idx + 1) * offset]
                tag_preds = tag_preds.softmax(dim=-1).argmax(dim=-1)

                prediction, reference = [], []
                for j in range(preds[i].shape[0]):
                    if tag_preds[j] == 0:
                        prediction.append("O")
                    elif tag_preds[j] == 1:
                        prediction.append("I-" + tag)
                    else:
                        prediction.append("B-" + tag)
                    reference.append(labels_words[i][tag][j])

                predictions[tag].append(prediction)
                references[tag].append(reference)
        predictions = [p for tag_preds in predictions.values() for p in tag_preds]
        references = [r for tag_refs in references.values() for r in tag_refs]
        # seqeval_results = self.seqeval.compute(predictions=predictions, references=references)
        # results = classification_report(references, predictions, output_dict=True)
        results = classification_report(references, predictions, output_dict=True)
        return results


class NERModule(L.LightningModule):
    def __init__(self, lm: nn.Module, lm_output_size: int, label2tag: int, iob_tags: bool, learning_rate: float = 2e-5):
        super().__init__()
        self.lm = lm
        self.lm.train()
        self.label2tag = label2tag
        self.learning_rate = learning_rate
        self.iob_tags = iob_tags
        self.offset = 3 if iob_tags else 2
        self.num_tags = len(label2tag.keys())
        self.num_labels = self.num_tags * self.offset
        self.classifier = nn.Linear(lm_output_size, self.num_labels)
        self.lm_output_size = lm_output_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = NEREval(iob_tags=self.iob_tags, tag2label={v: k for k, v in self.label2tag.items()})

    def exclude_padding_and_special_tokens(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1, self.num_labels)
        label_mask = labels[:, 0] != -100  # exclude padding and special tokens
        logits = logits[label_mask]
        labels = labels[label_mask]
        return logits, labels

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = []
        for i in range(self.num_tags):
            loss.append(
                self.loss_fn(
                    logits[:, i * self.offset : (i + 1) * self.offset], labels[:, i * self.offset : (i + 1) * self.offset]
                )
            )
        loss = torch.stack(loss).mean()
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        logits, labels = self.exclude_padding_and_special_tokens(logits, labels)

        loss = self.compute_loss(logits, labels)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        logits, labels = self.exclude_padding_and_special_tokens(logits, labels)
        loss = self.compute_loss(logits, labels)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        word_to_token = batch["word_to_token"]
        labels_words = batch["labels_words"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        all_preds = []
        all_labels_words = {}
        for i in range(input_ids.shape[0]):
            for j in range(len(word_to_token[i])):
                if word_to_token[i][j] is not None:
                    pred = logits[i, word_to_token[i][j][0]]  # use only the first token to predict the word-level entity
                    all_preds.append(pred)
            if i != input_ids.shape[0] - 1:  # add sep token between chunks
                sep_pred = torch.zeros(self.num_labels, device=pred.device)
                for k in range(self.num_tags):
                    sep_pred[k * self.offset] = 1
                all_preds.append(sep_pred)
            for tag in labels_words[i].keys():
                if tag not in all_labels_words:
                    all_labels_words[tag] = []
                lw = labels_words[i][tag]
                if i != input_ids.shape[0] - 1:  # add O tag between chunks
                    lw.append("O")
                all_labels_words[tag].extend(lw)
        all_preds = torch.stack(all_preds)
        self.metric.update(all_preds, all_labels_words)

    def on_test_epoch_end(self):
        results = self.metric.compute()
        for label in results.keys():
            for metric, metric_value in results[label].items():
                self.log(f"test_{label}_{metric}", metric_value, on_epoch=True, sync_dist=True)
        self.metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class RobertaNER(RobertaForTokenClassification):
    """
    A custom HF model that can load any backbone supported by AutoModel,
    then adds a linear classification head for multi-label NER.
    """

    def __init__(self, config):
        super().__init__(config)
        # 1) Instantiate the base model from config:
        self.base_model = AutoModel.from_config(config)

        # 2) Add your classifier head
        #    (requires config.hidden_size and config.num_labels to be set properly)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # HF's recommended approach for final init steps
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 3) Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # The base AutoModel typically returns a last_hidden_state as outputs[0]
        # But check the docs if you're using a model that returns a different format
        sequence_output = outputs[0]  # shape: (batch_size, seq_len, hidden_size)

        # 4) Apply classification head
        logits = self.classifier(sequence_output)

        # 5) Compute (optional) training loss
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"loss": loss, "logits": logits}


class BertNER(BertForTokenClassification):
    """
    A custom HF model that can load any backbone supported by AutoModel,
    then adds a linear classification head for multi-label NER.
    """

    def __init__(self, config):
        super().__init__(config)
        # 1) Instantiate the base model from config:
        self.base_model = AutoModel.from_config(config)

        # 2) Add your classifier head
        #    (requires config.hidden_size and config.num_labels to be set properly)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # HF's recommended approach for final init steps
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 3) Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # The base AutoModel typically returns a last_hidden_state as outputs[0]
        # But check the docs if you're using a model that returns a different format
        sequence_output = outputs[0]  # shape: (batch_size, seq_len, hidden_size)

        # 4) Apply classification head
        logits = self.classifier(sequence_output)

        # 5) Compute (optional) training loss
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"loss": loss, "logits": logits}


class XLMRobertaNER(XLMRobertaForTokenClassification):
    """
    A custom HF model that can load any backbone supported by AutoModel,
    then adds a linear classification head for multi-label NER.
    """

    def __init__(self, config):
        super().__init__(config)
        # 1) Instantiate the base model from config:
        self.base_model = AutoModel.from_config(config)

        # 2) Add your classifier head
        #    (requires config.hidden_size and config.num_labels to be set properly)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # HF's recommended approach for final init steps
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 3) Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # The base AutoModel typically returns a last_hidden_state as outputs[0]
        # But check the docs if you're using a model that returns a different format
        sequence_output = outputs[0]  # shape: (batch_size, seq_len, hidden_size)

        # 4) Apply classification head
        logits = self.classifier(sequence_output)

        # 5) Compute (optional) training loss
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="NER trainer")
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    argparser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    argparser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    argparser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    argparser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs for training")
    argparser.add_argument(
        "--root_path",
        type=str,
        default="T://laupodteam/AIOS/Bram/notebooks/code_dev/CardioNER.nl/assets",
        help="Root path for the dataset",
    )
    argparser.add_argument("--lang", type=str, default="nl", help="Language of the dataset")
    argparser.add_argument("--model_name", type=str, default="CLTL/MedRoBERTa.nl", help="Name of the pre-trained model")
    argparser.add_argument("--devices", type=int, nargs="+", default=[4], help="List of devices to use for training")
    argparser.add_argument("--use_cpu", action="store_true", help="Flag to use CPU for training")
    argparser.add_argument("--file_encoding", type=str, help="Important:encoding of train/val/test files. Check carefully.")
    argparser.add_argument("--with_suggestion", action="store_true", help="Use corpus with suggested annotations")
    argparser.add_argument("--use_iob_tags", action="store_true", help="Use IOB tags for labels")
    argparser.add_argument("--output_dir", type=str, default=".")

    args = argparser.parse_args()

    if args.file_encoding is None:
        import locale

        file_encoding = locale.getpreferredencoding(False)  # None
        print(f"WARNING: no file_encoding set, using system default: {file_encoding}")

    batch_size = args.batch_size
    patience = args.patience
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    root_path = args.root_path
    lang = args.lang
    model_name = args.model_name
    devices = args.devices
    use_cpu = args.use_cpu
    with_suggestion = args.with_suggestion
    output_dir = args.output_dir
    use_iob_tags = args.use_iob_tags

    tag2label = {
        "DISEASE": 0,
        "MEDICATION": 1,
        "PROCEDURE": 2,
        "SYMPTOM": 3,
    }
    label2tag = {v: k for k, v in tag2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

    max_len = model.config.max_position_embeddings
    if "Roberta" in model.config.architectures[0] or "Camembert" in model.config.architectures[0]:
        max_len = max_len - 2

    print(f"The maximum length: {max_len}")

    train = CardioCCC(
        root_path,
        "train",
        lang=lang,
        encoding=file_encoding,
        with_suggestion=with_suggestion,
        iob_tags=use_iob_tags,
        tag2label=tag2label,
    )
    val = CardioCCC(
        root_path,
        "validation",
        lang=lang,
        encoding=file_encoding,
        with_suggestion=with_suggestion,
        iob_tags=use_iob_tags,
        tag2label=tag2label,
    )
    test = CardioCCC(
        root_path,
        "test",
        lang=lang,
        encoding=file_encoding,
        with_suggestion=with_suggestion,
        iob_tags=use_iob_tags,
        tag2label=tag2label,
    )
    train = ChunkedCardioCCC(train, tokenizer, lang, tag2label=tag2label, iter_by_chunk=True, model_max_len=max_len)
    val = ChunkedCardioCCC(val, tokenizer, lang, tag2label=tag2label, iter_by_chunk=True, model_max_len=max_len)
    test = ChunkedCardioCCC(test, tokenizer, lang, tag2label=tag2label, iter_by_chunk=False, model_max_len=max_len)

    collate_fn_train = partial(collate_fn_chunked_bert, padding_value=tokenizer.pad_token_id)
    collate_fn_test = partial(collate_fn_chunked_bert_test, padding_value=tokenizer.pad_token_id)
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn_train, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=1, collate_fn=collate_fn_test, shuffle=False, num_workers=num_workers)

    module = NERModule(lm=model, lm_output_size=model.config.hidden_size, label2tag=label2tag, iob_tags=use_iob_tags, learning_rate=learning_rate)

    if torch.cuda.is_available() & use_cpu == False:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("medium")

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        ModelCheckpoint(monitor="val_loss", mode="min"),
    ]
    strategy = "ddp_find_unused_parameters_true" if len(devices) > 1 else "auto"
    strategy = "auto" if use_cpu else strategy  # ddp_spawn if use_cpu and not in notebook

    trainer = L.Trainer(
        default_root_dir=output_dir,
        callbacks=callbacks,
        accelerator="cpu" if use_cpu else "auto",
        devices="auto" if use_cpu else devices,
        max_epochs=max_epochs,
        strategy=strategy,
        precision="16-mixed" if isinstance(devices, list) or devices == "cuda" else "bf16",
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=module, dataloaders=test_loader)

    # save as huggingface compatible model
    ########################################
    # make transformer class
    base_config = AutoConfig.from_pretrained(model_name)
    base_config.num_labels = module.num_labels
    base_config.add_pooling_layer = False

    save_dir = os.path.join(output_dir, "hf")
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    if base_config.architectures[0].startswith("Roberta"):
        hf_model = RobertaNER(base_config)
        print(f"Storing as RobertaNER model in {save_dir}")
    elif base_config.architectures[0].startswith("Bert"):
        hf_model = BertNER(base_config)
        print(f"Storing as BertNER model in {save_dir}")
    elif base_config.architectures[0].startswith("XLMRoberta"):
        hf_model = XLMRobertaNER(base_config)
        print(f"Storing as XLMRobertaNER model in {save_dir}")

    hf_model.base_model.load_state_dict(module.to("cpu").lm.state_dict(), strict=False)
    hf_model.classifier.load_state_dict(module.to("cpu").classifier.state_dict())

    hf_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
