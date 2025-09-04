from os import environ
import spacy
import shutil

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from typing import List, Dict, Optional, Union
from collections import defaultdict
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import RobertaModel, RobertaForTokenClassification
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import torch
from transformers.utils import logging
from torchcrf import CRF
from utils import pretty_print_classifier

# Import custom model classes
try:
    from .modeling import TokenClassificationModelCRF, TokenClassificationModel
except ImportError:
    try:
        from modeling import TokenClassificationModelCRF, TokenClassificationModel
    except ImportError:
        print("Warning: Could not import custom model classes from modeling.py")
        TokenClassificationModelCRF = None
        TokenClassificationModel = None

logging.set_verbosity_debug()

from torch import nn

import evaluate
metric = evaluate.load("seqeval")

# TODO: add multihead CRF: https://github.com/ieeta-pt/Multi-Head-CRF/tree/master/src/model




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
            # num_labels = len(label2id)  # unused variable

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
        or_config.classifier_hidden_layers = classifier_hidden_layers
        or_config.classifier_dropout = classifier_dropout
        or_config.class_weights = class_weights

        self.classifier_hidden_layers= classifier_hidden_layers
        self.classifier_dropout = classifier_dropout
        self.class_weights = class_weights

        if use_crf:
            if TokenClassificationModelCRF is None:
                raise ImportError("TokenClassificationModelCRF could not be imported. Please ensure modeling.py is available.")
            print("USING CRF:", self.crf)
            base_model = RobertaForTokenClassification.from_pretrained(model, config=or_config)
            self.model = TokenClassificationModelCRF(or_config, base_model, freeze_backbone, classifier_hidden_layers, classifier_dropout)

            # Set up auto_map for trust_remote_code loading
            self.model.config.auto_map = {
                "AutoModelForTokenClassification": "modeling.TokenClassificationModelCRF"
            }
        else:
            if TokenClassificationModel is None:
                raise ImportError("TokenClassificationModel could not be imported. Please ensure modeling.py is available.")
            self.model = TokenClassificationModel.from_pretrained(model, config=or_config)

            # Set up auto_map for trust_remote_code loading
            self.model.config.auto_map = {
                "AutoModelForTokenClassification": "modeling.TokenClassificationModel"
            }

        # Mark that this is a custom model requiring trust_remote_code
        self.model.config.custom_model_type = "TokenClassificationModelCRF" if use_crf else "TokenClassificationModel"
        self.model.config.requires_trust_remote_code = True

        # optionally freeze the backbone
        if freeze_backbone:
            for p in self.model.roberta.parameters():
                p.requires_grad = False
            self.model.roberta.eval()

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


    def save_model(self, output_dir=None):
        """Custom method to save both the model architecture and state dict properly"""
        save_dir = output_dir or self.output_dir

        # Copy the modeling.py file to output_dir for trust_remote_code compatibility
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modeling_src = os.path.join(current_dir, 'modeling.py')
        modeling_dst = os.path.join(save_dir, 'modeling.py')

        # Ensure output directory exists
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(modeling_src):
            shutil.copyfile(modeling_src, modeling_dst)
            print(f"Successfully copied modeling.py to {modeling_dst}")

            # Verify the copy was successful
            if not os.path.exists(modeling_dst):
                raise ValueError(f"Failed to copy modeling.py to {modeling_dst}")
        else:
            raise ValueError(f"modeling.py not found at {modeling_src}. This file is required for custom models with trust_remote_code=True.")

        # Set model configuration
        self.model.config.architectures = [self.model.__class__.__name__]
        self.model.config.classifier_hidden_layers = self.classifier_hidden_layers
        self.model.config.classifier_dropout = self.classifier_dropout
        self.model.config.class_weights = self.class_weights

        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        print(f"Custom model saved successfully! To load this model, use:")
        print(f"AutoModelForTokenClassification.from_pretrained('{save_dir}', trust_remote_code=True)")
        print(f"Model saved to {save_dir}")


    def train(self,
        train_data: List[Dict],
        test_data: List[Dict],
        eval_data: List[Dict],
        profile: bool=False):

        if len(test_data)>0:
            _eval_data = test_data
        else:
            _eval_data = eval_data

        # Custom save function to properly handle non-PreTrainedModel models
        class CustomTrainer(Trainer):
            def __init__(self, parent_trainer, **kwargs):
                super().__init__(**kwargs)
                self.parent_trainer = parent_trainer

            def save_model(self, output_dir=None, _internal_call=False):
                self.parent_trainer.save_model(output_dir)

        trainer = CustomTrainer(
            parent_trainer=self,
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
        self.model = self.model.to(dtype=torch.bfloat16)
        try:
            trainer.save_model(self.output_dir)
            trainer.save_metrics(self.output_dir, metrics=metrics)
        except Exception as e:
            try:
                trainer.save_model('output')
                trainer.save_metrics('output', metrics=metrics)
                print(f"Saved to fallback directory 'output' due to error: {str(e)}")
            except Exception as e2:
                raise ValueError(f"Failed to save model and metrics: {str(e2)}")
        torch.cuda.empty_cache()
        return True
