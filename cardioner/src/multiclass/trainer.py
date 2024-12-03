from os import environ
import spacy

# Load a spaCy model for tokenization
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"

from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Tuple, Literal
from collections import defaultdict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from transformers.utils import logging
logging.set_verbosity_debug()

import evaluate
metric = evaluate.load("seqeval")

# https://huggingface.co/learn/nlp-course/en/chapter7/2

class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        # Call the superclass method to process the batch
        batch = super().__call__(features)
        
        # Retrieve input_ids and labels from the batch
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        def has_nan_or_inf(tensor):
            return torch.isnan(tensor).any() or torch.isinf(tensor).any()
    
        for key in batch:
            if has_nan_or_inf(batch[key]):
                print(f"Tensor {key} contains NaN or Inf values.")
                return None

        # Check for shape mismatch
            if input_ids.shape != labels.shape:
                raise ValueError(f"Shape mismatch: input_ids {input_ids.shape}, labels {labels.shape}")

            # Option 1: Adjust labels to match input_ids
            # This might involve padding or truncating labels
            #labels = self._adjust_labels(labels, input_ids.shape)
            #batch['labels'] = labels
            
            # Option 2: Skip this batch
   
            # Option 3: Raise an exception
            # raise ValueError("Mismatch between input_ids and labels shapes")
        
        return batch


    def _adjust_labels(self, labels, target_shape):
        """
        Adjusts the labels tensor to match the target_shape.
        This function pads or truncates the labels tensor.
        """
        # Get current shape
        current_shape = labels.shape
        
        # If labels are shorter, pad them
        if current_shape[1] < target_shape[1]:
            padding_length = target_shape[1] - current_shape[1]
            padding = torch.full((current_shape[0], padding_length), self.label_pad_token_id, dtype=labels.dtype)
            labels = torch.cat([labels, padding], dim=1)
        
        # If labels are longer, truncate them
        elif current_shape[1] > target_shape[1]:
            labels = labels[:, :target_shape[1]]
        
        return labels
    
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
                add_prefix_space=True, model_max_length=max_length, padding="max_length", truncation=True)
        else:
            self.tokenizer = tokenizer

        self.tokenizer.model_max_length = max_length
        self.data_collator = CustomDataCollatorForTokenClassification(tokenizer=self.tokenizer, max_length=max_length, 
                                                                padding="max_length", label_pad_token_id=-100)
        self.model = AutoModelForTokenClassification.from_pretrained(model, id2label=self.id2label, label2id=self.label2id, 
                                                                     num_labels=len(self.label2id))

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
        predictions = np.argmax(logits, axis=-1)

        # Access the id2label mapping
        id2label = self.id2label  # Dictionary mapping IDs to labels

        # Remove ignored index (special tokens) and convert to label names
        true_labels = [
            [id2label[l] for l in label if l != -100]
            for label in labels
        ]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }


    def train(self, train_data: List[Dict], eval_data: List[Dict], profile: bool=False):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
        )
        if profile:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                trainer.train()
        else:
            trainer.train()
        metrics = trainer.evaluate()
        try:
            trainer.save_model(r""+self.output_dir)
            trainer.save_metrics(r""+self.output_dir, metrics=metrics)
        except:
            print("Failed to save model and metrics")
        torch.cuda.empty_cache()
        return True