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
from transformers import TrainingArguments, Trainer, PreTrainedTokenizerBase
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from transformers.utils import logging
logging.set_verbosity_debug()

from utils import pretty_print_classifier
import evaluate
metric = evaluate.load("seqeval")

@dataclass
class MultiLabelDataCollatorForTokenClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = "max_length"
    max_length: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features, *args, **kwargs):
        labels = [feature.pop('labels') for feature in features]
        # Remove unnecessary keys if needed

        # Pad the inputs using the tokenizer's pad method
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert labels to tensors
        labels_tensors = [torch.tensor(label, dtype=torch.float) for label in labels]
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_tensors,
            batch_first=True,
            padding_value=self.label_pad_token_id
        )

        # Ensure labels are padded to max_seq_length
        max_seq_length = batch['input_ids'].shape[1]
        if padded_labels.shape[1] < max_seq_length:
            padding_size = max_seq_length - padded_labels.shape[1]
            padding = torch.full(
                (len(labels), padding_size, padded_labels.shape[2]),
                fill_value=self.label_pad_token_id,
                dtype=torch.float
            )
            padded_labels = torch.cat([padded_labels, padding], dim=1)
        elif padded_labels.shape[1] > max_seq_length:
            padded_labels = padded_labels[:, :max_seq_length, :]

        batch['labels'] = padded_labels

        return batch

class MultiLabelTokenClassificationModel(nn.Module): #AutoModelForTokenClassification):
    def __init__(self, config, base_model, freeze_backbone=False, classifier_hidden_layers=None,
        classifier_dropout=0.1, class_weights=None):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = base_model

        print(f"Creating model with freeze_backbone={freeze_backbone},\
                classifier_hidden_layers={classifier_hidden_layers},\
                classifier_dropout={classifier_dropout}")

        # Access custom attributes correctly
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
        #self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self._build_classifier_head(classifier_hidden_layers, classifier_dropout)

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
            if self.class_weights is not None:
                # Move class_weights to the same device as the model
                weights = self.class_weights.to(labels.device)
            else:
                weights = None
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=weights) # potentially weight=.. is better
            mask = (labels != -100).float()
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss_tensor = loss_fct(logits, labels.float())
            loss = (loss_tensor * mask).sum() / mask.sum()

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )

class MultiLabelTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.FloatTensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)
            logging.info(f"Using multi-label classification with class weights", class_weights)
        self.loss_fct = nn.BCEWithLogitsLoss(weight=class_weights, reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # # Compute the loss tensor with reduction='none'
        loss_tensor = self.loss_fct(logits.view(-1, model.num_labels),
                                     labels.view(-1, model.num_labels))
        # Apply mask to ignore padding (-100 labels)
        mask = (labels.view(-1, model.num_labels) != -100).float()
        loss_tensor = loss_tensor * mask

        # Reduce the loss tensor to a scalar
        loss = loss_tensor.sum() / mask.sum()

        return (loss, outputs) if return_outputs else loss

class ModelTrainer():
    def __init__(self,
                 label2id: Dict[str, int],
                 id2label: Dict[int, str],
                 tokenizer=None,
                 model: str='CLTL/MedRoBERTa.nl',
                 batch_size: int=48,
                 max_length: int=514,
                 learning_rate: float=1e-4,
                 weight_decay: float=0.001,
                 gradient_accumulation_steps: int=1,
                 num_train_epochs: int=20,
                 output_dir: str="../../output",
                 hf_token: str=None,
                 freeze_backbone: bool=False,
                 classifier_hidden_layers: tuple|None=None,
                 classifier_dropout: float=0.1,
                 class_weights: List[float]|None = None
    ):
        self.label2id = label2id
        self.id2label = id2label
        self.output_dir = output_dir

        self.train_kwargs = {
            'run_name': 'CardioNER',
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'weight_decay': weight_decay,
            'eval_strategy':'epoch',
            'save_strategy': 'best',
            'metric_for_best_model': 'f1_macro',
            'save_total_limit': 1,
            'report_to': 'tensorboard',
            'use_cpu': False,
            'fp16': True,
            'logging_dir': f"{output_dir}/logs",
            'logging_strategy': 'steps',
            'logging_steps': 256,
        }

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model,
                add_prefix_space=True,
                model_max_length=max_length,
                padding=False,
                truncation=False,
                token=hf_token)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        self.data_collator = MultiLabelDataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=max_length,
            label_pad_token_id=-100
        )
        or_config = AutoConfig.from_pretrained(model, hf_token=hf_token, return_unused_kwargs=False)
        or_config.num_labels=len(self.label2id)
        or_config.id2label=self.id2label
        or_config.label2id=self.label2id
        or_config.hidden_dropout_prob=0.1

        base_model = AutoModel.from_pretrained(model, token=hf_token)

        self.model = MultiLabelTokenClassificationModel(config=or_config,
                                                        base_model=base_model,
                                                        freeze_backbone=freeze_backbone,
                                                        classifier_hidden_layers=classifier_hidden_layers,
                                                        classifier_dropout=classifier_dropout,
                                                        class_weights=class_weights)

        print("Tokenizer max length:", self.tokenizer.model_max_length)
        print("Model max position embeddings:", self.model.config.max_position_embeddings)
        print("Number of labels:", len(self.label2id))
        print("Labels:", self.label2id)
        print("id2label:", self.id2label)
        print("Model config:", self.model.config)
        print("Head only fine-tuning:", freeze_backbone)
        print("Classifier architecture:", pretty_print_classifier(self.model.classifier))

        self.args = TrainingArguments(
            output_dir=output_dir,
            **self.train_kwargs
        )

    def compute_seqeval_metrics(self, eval_preds):
        logits, labels = eval_preds
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int().numpy()

        # we only consider the non-ambiguous labels
        idcs = np.argwhere(labels.sum(axis=-1)>0)
        labels = np.argmax(labels[idcs[:,0]], axis=-1)
        preds = np.argmax(preds[idcs[:,0]], axis=-1)

        # Access the id2label mapping
        id2label = self.id2label  # Dictionary mapping IDs to labels

        # Remove ignored index (special tokens) and convert to label names
        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]

        try:
            true_predictions = [
                [id2label[p] for (p, l) in zip(preds, label) if l != -100]
                for preds, label in zip(preds, labels)
            ]

            all_metrics = metric.compute(predictions=true_predictions,
                                references=true_labels)
            return all_metrics
        except Exception as e:
            print(f"Seqeval metrics failed: {e}. \n True labels sample: {true_labels[0]} \n Predictions sample: {true_predictions[0]}")
            return {}


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

        res_dict = {
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'rauc_micro': roc_auc_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'rauc_macro': roc_auc_macro,
        }

        precision_all = precision_score(labels, preds, average=None, zero_division=0)
        recall_all = recall_score(labels, preds, average=None, zero_division=0)
        f1_all = f1_score(labels, preds, average=None, zero_division=0)
        roc_auc_all = roc_auc_score(labels, preds, average=None, multi_class='ovr')

        precision_dict = defaultdict(float)
        recall_dict = defaultdict(float)
        f1_dict = defaultdict(float)
        roc_auc_dict = defaultdict(float)

        for k,v in self.id2label.items():
            precision_dict[f"precision_{v}"] = precision_all[k]
            recall_dict[f"recall_{v}"] = recall_all[k]
            f1_dict[f"f1_{v}"] = f1_all[k]
            roc_auc_dict[f"roc_auc_{v}"] = roc_auc_all[k]

        res_dict.update(precision_dict)
        res_dict.update(recall_dict)
        res_dict.update(f1_dict)
        res_dict.update(roc_auc_dict)

        # ADD metrics from seqeval
        #seq_eval = {f'SEQ_{k}':v for k,v in self.compute_seqeval_metrics(eval_preds).items()}
        #res_dict.update(seq_eval)
        return res_dict

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
