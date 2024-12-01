from os import environ
import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Literal
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AutoConfig, DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from transformers.utils import logging
logging.set_verbosity_debug()

import evaluate
metric = evaluate.load("seqeval")

# https://huggingface.co/learn/nlp-course/en/chapter7/2
@dataclass
class MultiLabelDataCollatorForTokenClassification:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = "max_length"
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature.pop('labels') for feature in features]
        # Remove keys not expected by the model
        for feature in features:
            keys_to_remove = ['id2label', 'tags', 'label2id', 'gid', 'batch', 'tokens', 'id']
            for key in keys_to_remove:
                feature.pop(key, None)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

        max_seq_length = batch['input_ids'].shape[1]
        batch_size = len(labels)
        num_labels = len(labels[0][0])

        # Initialize padded labels tensor
        padded_labels = torch.full(
            (batch_size, max_seq_length, num_labels),
            self.label_pad_token_id,
            dtype=torch.long
        )

        for i, label in enumerate(labels):
            seq_length = len(label)
            padded_labels[i, :seq_length, :] = torch.tensor(label, dtype=torch.long)

        batch['labels'] = padded_labels

        #print(f"Input IDs shape: {batch['input_ids'].shape}")
        #print(f"Labels shape: {batch['labels'].shape}")

        return batch

class MultiLabelTokenClassificationModel(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = AutoModel(config)  # Use the RobertaModel backbone
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss here if necessary
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            mask = (labels != -100).float()
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss_tensor = loss_fct(logits, labels.float())
            loss = (loss_tensor * mask).sum() / mask.sum()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        logits = model(**inputs)['logits']
        # Define BCEWithLogitsLoss here
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')  # Adjust reduction as needed
        mask = (labels != -100).float()
        labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
        loss_tensor = loss_fct(logits, labels.float())
        loss = (loss_tensor * mask).sum() / mask.sum()  # Mask invalid positions
        return (loss, logits) if return_outputs else loss

class ModelTrainer():
    def __init__(self,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 tokenizer=None,
                 model: str='CLTL/MedRoBERTa.nl',
                 batch_size: int=48,
                 max_length: int=514,
                 learning_rate: float=1e-4,
                 weight_decay: float=0.01,
                 num_train_epochs: int=3,
                 output_dir: str="../../output"
    ):
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        self.train_kwargs = {
            'run_name': 'CardioNER',
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'weight_decay': weight_decay,
            'eval_strategy':'epoch',
            'save_strategy': 'epoch',
            'save_total_limit': 1,
            'report_to': 'tensorboard',
            'use_cpu': False,
            'logging_dir': f"{output_dir}/logs",
            'logging_strategy': 'steps',
            'logging_steps': 256,
        }

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=False)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        self.data_collator = MultiLabelDataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=max_length,
            label_pad_token_id=-100
        )
        config = AutoConfig.from_pretrained(model, num_labels=len(self.label2id),
            id2label=self.id2label, label2id=self.label2id, hidden_dropout_prob=0.1)
        self.model = MultiLabelTokenClassificationModel.from_pretrained(model, config=config)

        print("Tokenizer max length:", self.tokenizer.model_max_length)
        print("Model max position embeddings:", self.model.config.max_position_embeddings)
        print("Number of labels:", len(self.label2id))
        print("Labels:", self.label2id)
        print("id2lagel:", self.id2label)
        print("Model config:", self.model.config)


        self.args = TrainingArguments(
            output_dir=r""+output_dir,
            **self.train_kwargs
        )

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int().numpy()
        labels = labels.reshape(-1, labels.shape[-1])
        preds = preds.reshape(-1, preds.shape[-1])


        print("Labels shape:", labels.shape)
        print("Preds shape:", preds.shape)

        # Exclude padded tokens
        #mask = (labels.sum(axis=1) != -100 * labels.shape[1])
        mask = ~np.all(labels == -100, axis=1)
        labels = labels[mask]
        preds = preds[mask]

        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
        recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
        roc_auc_micro = roc_auc_score(labels, preds, average='micro', multi_class='ovr')

        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        roc_auc_macro = roc_auc_score(labels, preds, average='macro', multi_class='ovr')

        return {
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'rauc_micro': roc_auc_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'rauc_macro': roc_auc_macro,
        }

    def train(self, train_data: List[Dict], eval_data: List[Dict]):
        trainer = MultiLabelTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )
        trainer.train()
        metrics = trainer.evaluate()
        try:
            trainer.save_model(r""+self.output_dir)
            trainer.save_metrics(r""+self.output_dir, metrics=metrics)
        except:
            print("Failed to save model and metrics")
        torch.cuda.empty_cache()
        return True
