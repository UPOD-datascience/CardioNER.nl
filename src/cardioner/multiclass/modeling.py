from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class TokenClassificationModelCRF(PreTrainedModel):
    """
    Custom token classification model with CRF layer and configurable classifier head.
    This model can be loaded with trust_remote_code=True for HuggingFace Hub compatibility.
    """

    def __init__(self, config, base_model=None, freeze_backbone=False,
                 classifier_hidden_layers=None, classifier_dropout=0.1):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels + 1  # Extra label to account for the -100 padding label
        self.pad_label = self.num_labels + 1

        # If base_model is not provided, load it from config
        if base_model is None:
            from transformers import RobertaForTokenClassification
            roberta_model = RobertaForTokenClassification.from_pretrained(
                config.name_or_path or config._name_or_path,
                config=config
            )
            self.roberta = roberta_model.roberta
        else:
            if hasattr(base_model, 'roberta'):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model

        self.lm_output_size = self.roberta.config.hidden_size

        # Store configuration for saving/loading
        self.config.freeze_backbone = freeze_backbone
        self.config.classifier_hidden_layers = classifier_hidden_layers
        self.config.classifier_dropout = classifier_dropout

        if freeze_backbone:
            print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")
            for param in self.roberta.parameters():
                param.requires_grad = False
            self.roberta.eval()
        else:
            print("+" * 30, "\n\n", "NOT Freezing backbone...", "+" * 30, "\n\n")

        self.roberta.train(not freeze_backbone)

        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.crf = CRF(self.num_labels, batch_first=True)

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
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        # Final classification layer
        layers.append(nn.Linear(input_size, self.num_labels))

        # Create sequential model
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # Emissions for CRF

        loss = None
        if labels is not None:
            # CRF calculates the log-likelihood of the correct sequence
            # We use a negative sign to convert it into a loss
            labels_long = labels.long()
            mask = attention_mask.bool() if attention_mask is not None else None
            loss = -self.crf(logits, labels_long, mask=mask, reduction='mean')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def device_info(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override from_pretrained to handle custom model loading"""
        config = kwargs.pop('config', None)
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Extract custom parameters from config if they exist
        freeze_backbone = getattr(config, 'freeze_backbone', False)
        classifier_hidden_layers = getattr(config, 'classifier_hidden_layers', None)
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        model = cls(
            config=config,
            freeze_backbone=freeze_backbone,
            classifier_hidden_layers=classifier_hidden_layers,
            classifier_dropout=classifier_dropout
        )

        # Load state dict if available
        try:
            state_dict = torch.load(
                f"{pretrained_model_name_or_path}/pytorch_model.bin",
                map_location="cpu"
            )
            model.load_state_dict(state_dict)
        except:
            # If loading fails, the model will be initialized with random weights
            print("Warning: Could not load pre-trained weights. Using randomly initialized model.")

        return model


class TokenClassificationModel(PreTrainedModel):
    """
    Custom token classification model with configurable classifier head (no CRF).
    This model can be loaded with trust_remote_code=True for HuggingFace Hub compatibility.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        # Initialize the roberta backbone
        from transformers import RobertaModel
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)

        # Get classifier configuration
        classifier_hidden_layers = getattr(config, "classifier_hidden_layers", None)
        classifier_dropout = getattr(config, "classifier_dropout", 0.1)

        # Build classifier head
        if classifier_hidden_layers is not None:
            # rebuild the MLP head
            in_size = config.hidden_size
            layers = []
            if classifier_hidden_layers:
                for h in classifier_hidden_layers:
                    layers += [
                        nn.Linear(in_size, h),
                        nn.ReLU(),
                        nn.Dropout(classifier_dropout)
                    ]
                    in_size = h
            layers.append(nn.Linear(in_size, config.num_labels))
            self.classifier = nn.Sequential(*layers)
        else:
            # Default single linear layer
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run inputs through the RoBERTa backbone
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                # Only keep active parts of the sequence
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)


def load_custom_cardioner_multiclass_model(model_path: str, device: str = "auto"):
    """
    Utility function to easily load a custom CardioNER multiclass model.

    Args:
        model_path: Path to the saved model directory
        device: Device to load model on ("auto", "cpu", "cuda", etc.)

    Returns:
        tuple: (model, tokenizer, config)
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    # Validate model directory
    import os
    required_files = ["config.json", "modeling.py", "pytorch_model.bin"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

    if missing_files:
        raise FileNotFoundError(f"Missing required files in {model_path}: {missing_files}")

    print(f"Loading custom CardioNER multiclass model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with trust_remote_code=True
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of labels: {model.num_labels}")

    return model, tokenizer, model.config


def validate_custom_multiclass_model_directory(model_path: str) -> dict:
    """
    Validate that a model directory contains all necessary files for custom multiclass model loading.

    Args:
        model_path: Path to the model directory

    Returns:
        dict: Validation results with status and details
    """
    import os
    import json

    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
        "model_info": {}
    }

    # Required files
    required_files = {
        "config.json": "Model configuration",
        "modeling.py": "Custom model class definition",
        "pytorch_model.bin": "Model weights"
    }

    # Optional files
    optional_files = {
        "tokenizer.json": "Tokenizer vocabulary",
        "tokenizer_config.json": "Tokenizer configuration",
        "training_args.json": "Training arguments"
    }

    # Check required files
    for filename, description in required_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Missing required file: {filename} - {description}")

    # Check optional files
    for filename, description in optional_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            validation_results["files_found"].append(f"{filename} ({description})")
        else:
            validation_results["warnings"].append(f"Missing optional file: {filename} - {description}")

    # Parse config if available
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            validation_results["model_info"]["num_labels"] = config.get("num_labels", "Unknown")
            validation_results["model_info"]["model_type"] = config.get("model_type", "Unknown")
            validation_results["model_info"]["has_auto_map"] = "auto_map" in config
            validation_results["model_info"]["classifier_hidden_layers"] = config.get("classifier_hidden_layers", None)
            validation_results["model_info"]["freeze_backbone"] = config.get("freeze_backbone", None)
            validation_results["model_info"]["use_crf"] = "TokenClassificationModelCRF" in str(config.get("architectures", []))

            if not config.get("auto_map"):
                validation_results["warnings"].append("No auto_map found in config - may not load correctly with trust_remote_code=True")

        except json.JSONDecodeError as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Invalid config.json: {str(e)}")

    # Check modeling.py content
    modeling_path = os.path.join(model_path, "modeling.py")
    if os.path.exists(modeling_path):
        try:
            with open(modeling_path, 'r') as f:
                content = f.read()

            required_classes = ["TokenClassificationModel", "TokenClassificationModelCRF"]
            missing_classes = [cls for cls in required_classes if cls not in content]

            if missing_classes:
                validation_results["valid"] = False
                validation_results["errors"].append(f"modeling.py missing required classes: {missing_classes}")

        except Exception as e:
            validation_results["warnings"].append(f"Could not read modeling.py: {str(e)}")

    return validation_results
