print("Importing libraries...")
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

print("Imported libraries..continuing to main")
torch.cuda.empty_cache()
gc.collect()

import argparse


#
def flatten_list(l):
    if not isinstance(l, list):
        return [l]
    return [item for sublist in l for item in flatten_list(sublist)]


def align_labels_to_text(text_encoding: Encoding, labels: list[dict], tag2label: dict):
    num_labels = len(tag2label.keys())
    text_labels = torch.zeros((text_encoding.input_ids.shape[1], num_labels))
    for label in labels:
        tag, start_idx, end_idx = label["tag"], int(label["start_span"]), int(label["end_span"])
        start_token_idx = text_encoding.char_to_token(start_idx)
        end_token_idx = text_encoding.char_to_token(end_idx - 1)
        try:
            text_labels[start_token_idx : end_token_idx + 1, tag2label[tag]] = 1
        except TypeError as e:
            print(f"Error: {e} for tag {tag}, start {start_idx}, end {end_idx}")
            raise TypeError("Check the labels/alignment/tokenizer")
    text_labels[~text_labels[:, 1:].any(dim=1), 0] = 1  # Adding null class if no other label is present
    return text_labels


def split_text(text: str, tokenizer: PreTrainedTokenizer, max_seq_len: int):
    paragraphs = re.split(r"(\n\n)", text)
    paragraphs = ["".join(paragraphs[i : i + 2]) for i in range(0, len(paragraphs), 2)]
    for p_idx in range(len(paragraphs)):
        ids = tokenizer.encode(paragraphs[p_idx], add_special_tokens=True)
        if len(ids) > max_seq_len:
            lines = re.split((r"(\n)"), paragraphs[p_idx])
            lines = ["".join(lines[i : i + 2]) for i in range(0, len(lines), 2)]
            for l_idx in range(len(lines)):
                ids = tokenizer.encode(lines[l_idx], add_special_tokens=True)
                if len(ids) > max_seq_len:
                    sentences = re.split(r"([.!?]\s+)", lines[l_idx])
                    sentences = ["".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)]
                    for s_idx in range(len(sentences)):
                        ids = tokenizer.encode(sentences[s_idx], add_special_tokens=True)
                        if len(ids) > max_seq_len:
                            words = re.split(r"(\s+)", sentences[s_idx])
                            words = ["".join(words[i : i + 2]) for i in range(0, len(words), 2)]
                            sentences[s_idx] = words
                    lines[l_idx] = sentences
            paragraphs[p_idx] = lines
    splits = flatten_list(paragraphs)
    return splits


def get_tokens_indices(char_to_token_list: list[int], start_idx: int, end_idx: int):
    token_idx_list = [char_to_token_list[i] for i in range(start_idx, end_idx) if char_to_token_list[i] is not None]
    token_idx_list = [k for k, _ in itertools.groupby(token_idx_list)]
    return token_idx_list


def merge_splits_into_chunks(
    text: str,
    splits: list[str],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    labels: list[dict],
    tag2label: dict,
):
    encoding = tokenizer(text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    char_to_token_list = [encoding.char_to_token(i) for i in range(len(text))]
    text_ids = encoding.input_ids[0]
    text_label_ids = align_labels_to_text(encoding, labels, tag2label)
    assert len(text_ids) == len(text_label_ids)

    # Merge splits into chunks without exceeding max_seq_len
    start_chunk_idx, end_chunk_idx = 0, 0
    chunks = {"text": [], "input_ids": [], "label_ids": []}
    for i in range(len(splits) + 1):
        # TODO: optimize this
        if i < len(splits):
            # Compute the current chunk length after adding the next tokenized split
            sentence = splits[i]
            token_idx_list = get_tokens_indices(char_to_token_list, start_chunk_idx, end_chunk_idx + len(sentence))
            chunk_ids = text_ids[token_idx_list]
        if i == len(splits) or len(chunk_ids) > max_seq_len:
            # add previous splits as a chunk if current chunk exceeds max_seq_len or if the splits are finished
            token_idx_list = get_tokens_indices(char_to_token_list, start_chunk_idx, end_chunk_idx)
            chunk_ids = text_ids[token_idx_list]
            chunk_labels_ids = text_label_ids[token_idx_list]
            chunks["text"].append(text[start_chunk_idx:end_chunk_idx])
            chunks["input_ids"].append(chunk_ids)
            chunks["label_ids"].append(chunk_labels_ids)
            start_chunk_idx = end_chunk_idx
        end_chunk_idx += len(sentence)
    return chunks


def split_iob(label: dict, nlp: spacy.Language, orig_text: str):
    new_labels = []
    text = label["text"]
    doc = nlp(text)
    first_token = doc[0]
    second_token = None
    if len(doc) > 1:
        second_token = doc[1]

    start_span = int(label["start_span"])
    end_span = int(label["end_span"])
    b_end_span = start_span + len(first_token.text)
    b_label = {
        "tag": "B-" + label["tag"],
        "start_span": start_span,
        "end_span": b_end_span,
        "text": first_token.text,
    }
    new_labels.append(b_label)
    if second_token:
        i_start_span = start_span + second_token.idx
        i_label = {
            "tag": "I-" + label["tag"],
            "start_span": i_start_span,
            "end_span": end_span,
            "text": text[second_token.idx :],
        }
        new_labels.append(i_label)
        assert orig_text[i_start_span:end_span] == i_label["text"], f"{orig_text[i_start_span:end_span]} != {i_label['text']}"
    assert orig_text[start_span:b_end_span] == b_label["text"]
    return new_labels


class CardioCCC(Dataset):
    LABEL_FOLDERS = ["dis", "med", "symp", "proc"]

    def __init__(
        self,
        root_path: str,
        split: str,
        lang: str = "it",
        encoding: str = "UTF-8",
        with_suggestion: bool = False,
        iob_tags: bool = False,
    ):

        val_str = "2_validated_w_sugs" if with_suggestion else "1_validated_without_sugs"
        self.root_path = Path(root_path)
        self.split_file_names = json.load((self.root_path / "splits.json").open())[lang][split]["symp"]
        self.lang = lang
        batches = ["b1", "b2"] if lang != "ro" else ["b1"]
        self.annotations = []
        if iob_tags:
            nlp = spacy.blank(lang)
        for batch in batches:
            lang_path = self.root_path / batch / val_str / lang
            raw_annotations = []
            for label_folder in self.LABEL_FOLDERS:
                ann_path = lang_path / label_folder / "tsv"
                raw_annotations.append(pd.read_csv(next(ann_path.glob("*.tsv")), sep="\t", na_filter=False))
            raw_annotations = pd.concat(raw_annotations, axis=0)

            for group in raw_annotations.groupby("name"):
                if group[0] not in self.split_file_names:
                    continue
                file_name = group[0] + ".txt"
                text = (lang_path / "dis/txt" / file_name).read_text(encoding=encoding)
                labels = group[1].loc[:, ["tag", "start_span", "end_span", "text"]].to_dict(orient="records")
                if iob_tags:
                    labels = [x for label in labels for x in split_iob(label, nlp, text)]
                self.annotations.append({"text": text, "labels": labels})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]


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
        for i, item in enumerate(dataset):
            text, labels = item["text"], item["labels"]
            splits = split_text(text, tokenizer, model_max_len)
            chunks = merge_splits_into_chunks(text, splits, tokenizer, model_max_len, labels, self.tag2label)
            if iter_by_chunk:
                for i in range(len(chunks["text"])):
                    self.chunked_data.append(
                        {
                            "text": chunks["text"][i],
                            "input_ids": chunks["input_ids"][i],
                            "label_ids": chunks["label_ids"][i],
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
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class NEREval(Metric):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        metric_classes_dict = {"f1": F1Score, "precision": Precision, "recall": Recall, "accuracy": Accuracy}
        self.classification_metrics = ModuleDict(
            {
                k
                + (f"_{avg}" if avg != "none" else ""): v(task="multilabel", num_labels=num_labels, average=avg, zero_division=1)
                for k, v in metric_classes_dict.items()
                for avg in ["none", "micro", "macro"]
            }
        )

    def update(self, preds: Tensor, labels: Tensor) -> None:
        self.preds.append(preds)
        self.labels.append(labels)

    def compute(self):
        preds, labels = self.preds, self.labels
        if isinstance(preds, list):
            preds, labels = torch.cat(self.preds), torch.cat(self.labels)

        results = {}
        for metric_name, metric in self.classification_metrics.items():
            results[metric_name] = metric(preds, labels)
            metric.reset()
        return results


class NERModule(L.LightningModule):
    def __init__(self, 
                 lm: nn.Module, 
                 lm_output_size: int, 
                 label2tag: int, 
                 freeze_backbone: bool = False,
                 learning_rate: float = 2e-5):
        super().__init__()
        self.lm = lm
        if freeze_backbone:
            print("Freezing transformer backbone")
            for param in self.lm.parameters():
                param.requires_grad = False
            self.lm.eval()
        self.lm.train(not freeze_backbone)    
        self.label2tag = label2tag
        self.num_labels = len(label2tag.keys())
        self.classifier = nn.Linear(lm_output_size, self.num_labels)
        self.lm_output_size = lm_output_size
        self.metric = NEREval(num_labels=self.num_labels)
        self.learning_rate = learning_rate

    def exclude_padding_and_special_tokens(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1, self.num_labels)
        label_mask = labels[:, 0] != -100  # exclude padding and special tokens
        logits = logits[label_mask]
        labels = labels[label_mask]
        return logits, labels

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        logits, labels = self.exclude_padding_and_special_tokens(logits, labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        logits, labels = self.exclude_padding_and_special_tokens(logits, labels)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        preds = logits.sigmoid()
        self.metric.update(preds, labels)

    def on_validation_epoch_end(self):
        results = self.metric.compute()
        for k, v in results.items():
            if "micro" not in k and "macro" not in k:
                for i in range(self.num_labels):
                    self.log(f"val_{k}_class_{self.label2tag[i]}", v[i], on_epoch=True, sync_dist=True)
            else:
                self.log(f"val_{k}", v, on_epoch=True, sync_dist=True)
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        sequence_out = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(sequence_out)
        logits, labels = self.exclude_padding_and_special_tokens(logits, labels)
        preds = logits.sigmoid()
        self.metric.update(preds, labels)

    def on_test_epoch_end(self):
        results = self.metric.compute()
        new_results = {}
        for k, v in results.items():
            if "micro" not in k and "macro" not in k:
                for i in range(self.num_labels):
                    new_results[f"test_{k}_class_{self.label2tag[i]}"] = v[i].item()
                    self.log(f"test_{k}_class_{self.label2tag[i]}", v[i], on_epoch=True, sync_dist=True)
            else:
                new_results[f"test_{k}"] = v.item()
                self.log(f"test_{k}", v, on_epoch=True, sync_dist=True)
        self.metric.reset()
        return new_results

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
    argparser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    argparser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    argparser.add_argument("--freeze_backbone", action="store_true", help="Freeze the transformer backbone and train only the classifier head")
    argparser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    argparser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    argparser.add_argument("--max_epochs", type=int, default=30, help="Maximum number of epochs for training")
    argparser.add_argument(
        "--root_path",
        type=str,
        default="/path/to/cardioCCC",
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
    else:
        file_encoding = args.file_encoding

    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    patience = args.patience
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    root_path = args.root_path
    lang = args.lang
    model_name = args.model_name
    devices = args.devices
    use_cpu = args.use_cpu
    with_suggestion = args.with_suggestion
    output_dir = args.output_dir
    use_iob_tags = args.use_iob_tags

    if use_iob_tags:
        tag2label = {
            "O": 0,
            "B-DISEASE": 1,
            "I-DISEASE": 2,
            "B-MEDICATION": 3,
            "I-MEDICATION": 4,
            "B-PROCEDURE": 5,
            "I-PROCEDURE": 6,
            "B-SYMPTOM": 7,
            "I-SYMPTOM": 8,
        }
    else:
        tag2label = {
            "O": 0,
            "DISEASE": 1,
            "MEDICATION": 2,
            "PROCEDURE": 3,
            "SYMPTOM": 4,
        }
    label2tag = {v: k for k, v in tag2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False, use_fast=True)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

    max_len = model.config.max_position_embeddings
    print(f"Model arch: {model.config.architectures}")
    if "Roberta" in model.config.architectures[0] or "Camembert" in model.config.architectures[0]:
        max_len = max_len - 2

    print(f"The maximum length: {max_len}")

    train = CardioCCC(root_path, "train", lang, encoding=file_encoding, with_suggestion=with_suggestion, iob_tags=use_iob_tags)
    val = CardioCCC(root_path, "validation", lang, encoding=file_encoding, with_suggestion=with_suggestion, iob_tags=use_iob_tags)
    test = CardioCCC(root_path, "test", lang, encoding=file_encoding, with_suggestion=with_suggestion, iob_tags=use_iob_tags)

    train = ChunkedCardioCCC(train, tokenizer, lang, tag2label=tag2label, iter_by_chunk=True, model_max_len=max_len)
    val = ChunkedCardioCCC(val, tokenizer, lang, tag2label=tag2label, iter_by_chunk=True, model_max_len=max_len)
    test = ChunkedCardioCCC(test, tokenizer, lang, tag2label=tag2label, iter_by_chunk=True, model_max_len=max_len)

    collate_fn = partial(collate_fn_chunked_bert, padding_value=tokenizer.pad_token_id)
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers)

    module = NERModule(lm=model,
                       lm_output_size=model.config.hidden_size,
                       learning_rate=args.learning_rate,
                       freeze_backbone=args.freeze_backbone,
                       label2tag=label2tag)


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
        accumulate_grad_batches=accumulation_steps,
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
use_cpu
