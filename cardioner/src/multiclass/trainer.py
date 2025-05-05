from os import environ
import spacy

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Literal
from collections import defaultdict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModel
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoConfig
from transformers import RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from transformers.utils import logging
from torchcrf import CRF
from utils import pretty_print_classifier

logging.set_verbosity_debug()

from torch import nn

import evaluate
metric = evaluate.load("seqeval")

# TODO: add multihead CRF: https://github.com/ieeta-pt/Multi-Head-CRF/tree/master/src/model

class TokenClassificationModelCRF(nn.Module):
    # TODO: add class weights to python-crf..
    # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    def __init__(self, config, base_model, freeze_backbone=False, classifier_hidden_layers=None,
        classifier_dropout=0.1):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels + 1 # Extra label to acocunt for the -100 padding label
        self.pad_label = self.num_labels + 1
        self.roberta = base_model

        self.lm_output_size = self.roberta.config.hidden_size

        if freeze_backbone:
            print(30*"+", "\n\n", "Freezing backbone...", 30*"+", "\n\n")
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.roberta.eval()
        else:
            print(30*"+", "\n\n", "NOT Freezing backbone...", 30*"+", "\n\n")
        self.roberta.train(not freeze_backbone)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)

        self._build_classifier_head(classifier_hidden_layers,
            classifier_dropout)

    def _build_classifier_head(self, hidden_layers, dropout_rate):
        """
        Build a flexible classifier head with configurable hidden layers and dropout.

        Args:
            hidden_layers: Tuple of integers representing the number of neurons in each hidden layer.
                            None or empty tuple means a simple linear layer.
            dropout_rate: Dropout probability between layers
        """
        layers = []
        input_size = self.lm_output_size

        # If hidden_layers is None or empty, just create a simple linear layer
        if not hidden_layers:
            self.classifier = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(input_size, self.num_labels)
                    )
            return

        # Build MLP with specified hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            #layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Final classification layer
        layers.append(nn.Linear(input_size, self.num_labels))

        # Create sequential model
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # Emissions for CRF

        loss = None
        if labels is not None:
            # CRF calculates the log-likelihood of the correct sequence
            # We use a negative sign to convert it into a loss
            labels = labels.long()
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')

        # Return a TokenClassifierOutput for interoperability
        return TokenClassifierOutput(
             loss=loss,
             logits=logits,             # raw emissions (logits)
             hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
             attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )

    @property
    def device_info(self):
        return next(self.parameters()).device


class TokenClassificationModel(nn.Module):
    def __init__(self, config, base_model, freeze_backbone=False, classifier_hidden_layers=None,
        classifier_dropout=0.1, class_weights=None):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = base_model

        self.lm_output_size = self.roberta.config.hidden_size

        # Store class weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

        if freeze_backbone:
            print(30*"+", "\n\n", "Freezing backbone...", 30*"+", "\n\n")
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.roberta.eval()
        else:
            print(30*"+", "\n\n", "NOT Freezing backbone...", 30*"+", "\n\n")
        self.roberta.train(not freeze_backbone)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._build_classifier_head(classifier_hidden_layers,
            classifier_dropout)

    def _build_classifier_head(self, hidden_layers, dropout_rate):
        """
        Build a flexible classifier head with configurable hidden layers and dropout.

        Args:
            hidden_layers: Tuple of integers representing the number of neurons in each hidden layer.
                            None or empty tuple means a simple linear layer.
            dropout_rate: Dropout probability between layers
        """
        layers = []
        input_size = self.lm_output_size

        # If hidden_layers is None or empty, just create a simple linear layer
        if not hidden_layers:
            self.classifier = nn.Linear(input_size, self.num_labels)
            return

        # Build MLP with specified hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Final classification layer
        layers.append(nn.Linear(input_size, self.num_labels))

        # Create sequential model
        self.classifier = nn.Sequential(*layers)
        print(f"Created classifier head with architecture: {self.classifier}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # Run inputs through the RoBERTa backbone
        # TODO: see https://github.com/huggingface/transformers/pull/35875/files#diff-5707805d290617078f996faf1138de197fa813f78c0aa5ea497e73b5228f1103
        # drop 'num_items_in_batch' from **kwargs
        # modeling_roberta.py
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                # Move class_weights to the same device as the model
                weights = self.class_weights.to(labels.device)
            else:
                weights = None
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            if attention_mask is not None:
                # Only keep active parts of the sequence
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features, *args, **kwargs):
        # Call the superclass method to process the batch
        #
        for feature in features:
            if 'labels' in feature:
                lbls = feature['labels']
                # If lbls is one-hot encoded (e.g. [seq_length, num_labels] per example),
                # first convert to a tensor and then argmax:
                if torch.Tensor(lbls).ndim == 2:
                    lbls_tensor = torch.tensor(lbls, dtype=torch.float)
                    lbls = lbls_tensor.argmax(dim=-1).tolist()
                    lbls = [label if label != -100 else self.label_pad_token_id for label in lbls]
                    if len(lbls) < len(feature['input_ids']):
                        lbls = lbls + [self.label_pad_token_id] * (len(feature['input_ids']) - len(lbls))
                    # Now lbls is a simple list of integers
                else:
                    lbls = [label if label != -100 else self.label_pad_token_id for label in lbls]
                    lbls = lbls + [self.label_pad_token_id] * (len(feature['input_ids']) - len(lbls))

                if (-100 in lbls) & (self.label_pad_token_id!=-100):
                    print("Warning: -100 found in labels after replacement!")

                feature['labels'] = lbls

        # Now call the superclass, which will handle converting everything to tensors
        batch = super().__call__(features)
        return batch


class ModelTrainer():
    def __init__(self,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 tokenizer=None,
                 model: str='CLTL/MedRoBERTa.nl',
                 use_crf: bool=False,
                 batch_size: int=48,
                 max_length: int=514,
                 learning_rate: float=1e-4,
                 weight_decay: float=0.001,
                 num_train_epochs: int=3,
                 output_dir: str="../../output",
                 hf_token: str=None,
                 freeze_backbone: bool=False,
                 gradient_accumulation_steps: int=1,
                 classifier_hidden_layers: tuple|None=None,
                 classifier_dropout: float=0.1,
                 class_weights: List[float]|None = None
    ):
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        self.train_kwargs = {
            'run_name': 'CardioNER',
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
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
        self.crf=use_crf

        if use_crf:
            self.pad_token_id = len(label2id) + 1
            num_labels = len(label2id) + 1
            id2label[self.pad_token_id]='PADDING'
            label2id['PADDING'] = self.pad_token_id
        else:
            self.pad_token_id  = -100
            num_labels = len(label2id)

        if tokenizer is None:
            print("LOADING TOKENIZER")
            self.tokenizer = AutoTokenizer.from_pretrained(model,
                add_prefix_space=True, model_max_length=max_length, padding="max_length", truncation=True, token=hf_token)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        self.data_collator = CustomDataCollatorForTokenClassification(tokenizer=self.tokenizer, max_length=max_length,
                                                                padding="max_length", label_pad_token_id=self.pad_token_id )

        or_config = AutoConfig.from_pretrained(model, hf_token=hf_token, return_unused_kwargs=False)
        or_config.num_labels=len(self.label2id)
        or_config.id2label=self.id2label
        or_config.label2id=self.label2id
        or_config.hidden_dropout_prob=0.1

        if use_crf:
            print("USING CRF:", self.crf)
            base_model = RobertaModel.from_pretrained(model, config=or_config)
            self.model = TokenClassificationModelCRF(or_config, base_model, freeze_backbone, classifier_hidden_layers, classifier_dropout)
        else:
            base_model = RobertaModel.from_pretrained(model, config=or_config)
            self.model = TokenClassificationModel(or_config, base_model, freeze_backbone, classifier_hidden_layers, classifier_dropout, class_weights)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Model:", type(self.model))
        print("Device:", self.device)
        print("Tokenizer max length:", self.tokenizer.model_max_length)
        print("Model max position embeddings:", self.model.config.max_position_embeddings)
        print("Number of labels:", len(self.label2id))
        print("Labels:", self.label2id)
        print("id2label:", self.id2label)
        print("Model config:", self.model.config)
        print("Classifier architecture:", pretty_print_classifier(self.model.classifier))

        self.args = TrainingArguments(
            output_dir=output_dir,
            **self.train_kwargs
        )

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        if self.crf:
            mask = labels != self.pad_token_id
            mask[:, 0] = True

            # self.model.device_info()
            emissions_torch = torch.from_numpy(logits).float().to(self.device)
            mask_torch = torch.from_numpy(mask).bool().to(self.device)

            predictions = self.model.crf.decode(emissions=emissions_torch, mask=mask_torch)
        else:
            predictions = np.argmax(logits, -1)

        # Access the id2label mapping
        id2label = self.id2label  # Dictionary mapping IDs to labels

        # Remove ignored index (special tokens) and convert to label names
        true_labels = [
            [id2label[l] for l in label if l != self.pad_token_id]
            for label in labels
        ]

        try:
            true_predictions = [
                [id2label[p] for (p, l) in zip(prediction, label) if l != self.pad_token_id]
                for prediction, label in zip(predictions, labels)
            ]
        except Exception as e:
            print(f"Predictions: {predictions}")
            print(f"Labels: {labels}")

        all_metrics = metric.compute(predictions=true_predictions,
                        references=true_labels)
        return all_metrics


    def train(self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool=False):

        if len(test_data)>0:
            _eval_data = test_data
        else:
            _eval_data = eval_data

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=_eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )

        if profile:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=False,
                profile_memory=True,
                with_stack=True
            ) as prof:
                trainer.train()
        else:
            trainer.train()

        # TODO: if there is a test set and evaluation set, evaluate on the eval set
        metrics = trainer.evaluate(eval_dataset=eval_data)
        try:
            trainer.save_model(self.output_dir)
            trainer.save_metrics(self.output_dir, metrics=metrics)
        except:
            trainer.save_model('output')
            trainer.save_metrics('output', metrics=metrics)
            print("Failed to save model and metrics")
        torch.cuda.empty_cache()
        return True
