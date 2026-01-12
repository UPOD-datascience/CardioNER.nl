from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
from itertools import islice


# Large negative number for masking impossible transitions
LARGE_NEGATIVE_NUMBER = -1e9
NUM_PER_LAYER = 16


class MultiHeadCRFConfig(PretrainedConfig):
    """
    Configuration class for Multi-Head CRF models.

    Args:
        entity_types: List of entity type names (e.g., ["DRUG", "DISEASE", "SYMPTOM"])
        number_of_layers_per_head: Number of dense layers per head before classification
        crf_reduction: Reduction mode for CRF loss ("mean", "sum", "token_mean", "none")
        freeze_backbone: Whether to freeze the transformer backbone
        num_frozen_encoders: Number of encoder layers to freeze (from bottom)
        classifier_dropout: Dropout rate for classifier heads
        **kwargs: Additional arguments passed to PretrainedConfig
    """
    model_type = "multihead-crf-tagger"

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        number_of_layers_per_head: int = 1,
        crf_reduction: str = "mean",
        freeze_backbone: bool = False,
        num_frozen_encoders: int = 0,
        classifier_dropout: float = 0.1,
        classifier_hidden_layers: Optional[Tuple] = None,
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        self.entity_types = entity_types or []
        self.number_of_layers_per_head = number_of_layers_per_head
        self.crf_reduction = crf_reduction
        self.freeze_backbone = freeze_backbone
        self.num_frozen_encoders = num_frozen_encoders
        self.classifier_dropout = classifier_dropout
        self.classifier_hidden_layers = classifier_hidden_layers
        self.class_weights = class_weights
        super().__init__(**kwargs)


class MultiHeadCRF(nn.Module):
    """
    Custom CRF implementation with BIO transition masking.

    This CRF implementation includes:
    - Proper initialization of transition parameters
    - Masking of impossible BIO transitions (e.g., O -> I is invalid)
    - Viterbi decoding for inference

    Args:
        num_tags: Number of tags (typically 3 for BIO: O, B, I)
        batch_first: Whether batch dimension is first
    """

    def __init__(self, num_tags: int, batch_first: bool = True) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()
        self.mask_impossible_transitions()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters uniformly between -0.1 and 0.1."""
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def mask_impossible_transitions(self) -> None:
        """
        Set impossible BIO transitions to large negative values.

        For standard BIO tagging with tags [O=0, B=1, I=2]:
        - Cannot start with I tag
        - Cannot transition from O to I
        """
        with torch.no_grad():
            # Assuming BIO scheme: O=0, B=1, I=2
            # Cannot start with I
            if self.num_tags > 2:
                self.start_transitions[2] = LARGE_NEGATIVE_NUMBER
                # Cannot go from O to I
                self.transitions[0][2] = LARGE_NEGATIVE_NUMBER

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute the negative log likelihood of a sequence of tags given emission scores.

        Args:
            emissions: Emission scores (batch_size, seq_length, num_tags) if batch_first
            tags: Gold tag sequence (batch_size, seq_length) if batch_first
            mask: Mask tensor (batch_size, seq_length) if batch_first
            reduction: Loss reduction mode ("none", "sum", "mean", "token_mean")

        Returns:
            Negative log likelihood loss
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator
        nllh = -llh

        if reduction == 'none':
            return nllh
        if reduction == 'sum':
            return nllh.sum()
        if reduction == 'mean':
            return nllh.mean()
        assert reduction == 'token_mean'
        return nllh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores
            mask: Mask tensor

        Returns:
            List of best tag sequences for each batch
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor,
                        mask: torch.Tensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        score += self.end_transitions

        # Trace back
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


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


class TokenClassificationModelMultiHeadCRF(PreTrainedModel):
    """
    Multi-Head CRF model for token classification with multiple entity types.

    Each entity type gets its own classification head and CRF layer, allowing
    for independent BIO tagging per entity type. This is useful for scenarios
    where entities can overlap or when different entity types have different
    transition patterns.

    Args:
        config: MultiHeadCRFConfig or compatible config with entity_types
        base_model: Optional pre-trained RoBERTa model
        freeze_backbone: Whether to freeze transformer weights
    """
    config_class = MultiHeadCRFConfig
    base_model_prefix = "roberta"
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, base_model=None, freeze_backbone=None):
        super().__init__(config)
        self.config = config

        # Get entity types from config
        self.entity_types = getattr(config, 'entity_types', [])
        if not self.entity_types:
            raise ValueError("entity_types must be provided in config")

        # Number of labels per head (typically 3 for BIO: O, B, I) + padding
        self.num_labels = config.num_labels
        self.num_labels_with_pad = self.num_labels + 1

        # Configuration parameters
        self.number_of_layers_per_head = getattr(config, 'number_of_layers_per_head', 1)
        self.crf_reduction = getattr(config, 'crf_reduction', 'mean')
        freeze_backbone = freeze_backbone if freeze_backbone is not None else getattr(config, 'freeze_backbone', False)
        self.num_frozen_encoders = getattr(config, 'num_frozen_encoders', 0)
        classifier_dropout = getattr(config, 'classifier_dropout', 0.1)

        # Initialize the transformer backbone
        if base_model is None:
            from transformers import RobertaModel
            self.roberta = RobertaModel(config, add_pooling_layer=False)
        else:
            if hasattr(base_model, 'roberta'):
                self.roberta = base_model.roberta
            else:
                self.roberta = base_model

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))

        # Create heads for each entity type
        print(f"Creating Multi-Head CRF with entity types: {sorted(self.entity_types)}")

        for entity_type in self.entity_types:
            # Dense layers per head
            for i in range(self.number_of_layers_per_head):
                setattr(self, f"{entity_type}_dense_{i}",
                        nn.Linear(self.hidden_size, self.hidden_size))
                setattr(self, f"{entity_type}_dense_activation_{i}",
                        nn.GELU(approximate='none'))
                setattr(self, f"{entity_type}_dropout_{i}",
                        nn.Dropout(classifier_dropout))

            # Classifier and CRF per head
            setattr(self, f"{entity_type}_classifier",
                    nn.Linear(self.hidden_size, self.num_labels_with_pad))
            setattr(self, f"{entity_type}_crf",
                    MultiHeadCRF(num_tags=self.num_labels_with_pad, batch_first=True))

        # Handle freezing
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze transformer backbone parameters."""
        print("+" * 30, "\n\n", "Freezing backbone...", "+" * 30, "\n\n")

        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        # Optionally freeze some encoder layers
        if self.num_frozen_encoders > 0:
            for _, param in islice(self.roberta.encoder.named_parameters(),
                                   self.num_frozen_encoders * NUM_PER_LAYER):
                param.requires_grad = False

    def reset_head_parameters(self):
        """Reset parameters for all heads (useful after loading pretrained weights)."""
        for entity_type in self.entity_types:
            for i in range(self.number_of_layers_per_head):
                getattr(self, f"{entity_type}_dense_{i}").reset_parameters()
            getattr(self, f"{entity_type}_classifier").reset_parameters()
            getattr(self, f"{entity_type}_crf").reset_parameters()
            getattr(self, f"{entity_type}_crf").mask_impossible_transitions()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Dict[str, torch.LongTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass through the multi-head CRF model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Dictionary mapping entity types to label tensors
                   e.g., {"DRUG": tensor, "DISEASE": tensor}
            ... other standard transformer arguments

        Returns:
            During training (labels provided):
                Tuple of (total_loss, logits_dict) where logits_dict maps entity types to logits
            During inference (no labels):
                List of prediction tensors, one per entity type (sorted alphabetically)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get transformer outputs
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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)  # (batch, seq_len, hidden)

        # Compute logits for each head
        logits = {}
        for entity_type in self.entity_types:
            head_output = sequence_output
            for i in range(self.number_of_layers_per_head):
                head_output = getattr(self, f"{entity_type}_dense_{i}")(head_output)
                head_output = getattr(self, f"{entity_type}_dense_activation_{i}")(head_output)
                head_output = getattr(self, f"{entity_type}_dropout_{i}")(head_output)
            logits[entity_type] = getattr(self, f"{entity_type}_classifier")(head_output)

        if labels is not None:
            # Training mode - compute CRF loss for each head
            losses = {}
            mask = attention_mask.bool() if attention_mask is not None else None

            for entity_type in self.entity_types:
                if entity_type in labels:
                    entity_labels = labels[entity_type].long()
                    crf = getattr(self, f"{entity_type}_crf")
                    # CRF returns negative log likelihood, we want to minimize it
                    losses[entity_type] = crf(
                        logits[entity_type],
                        entity_labels,
                        mask=mask,
                        reduction=self.crf_reduction
                    )

            # Sum losses from all heads
            total_loss = sum(losses.values())
            return total_loss, logits

        else:
            # Inference mode - decode each head
            predictions = {}
            mask = attention_mask.bool() if attention_mask is not None else None

            for entity_type in self.entity_types:
                crf = getattr(self, f"{entity_type}_crf")
                decoded = crf.decode(logits[entity_type], mask=mask)
                predictions[entity_type] = torch.tensor(decoded)

            # Return as list sorted by entity type for consistency
            return [predictions[ent] for ent in sorted(self.entity_types)]

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.roberta.set_input_embeddings(value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override from_pretrained to handle custom model loading."""
        config = kwargs.pop('config', None)
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        freeze_backbone = getattr(config, 'freeze_backbone', False)

        model = cls(config=config, freeze_backbone=freeze_backbone)

        # Load state dict if available
        import os
        weight_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        safetensors_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")

        try:
            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_file)
                model.load_state_dict(state_dict)
            elif os.path.exists(weight_file):
                state_dict = torch.load(weight_file, map_location="cpu")
                model.load_state_dict(state_dict)
            else:
                print("Warning: No pre-trained weights found. Using randomly initialized model.")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")

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


def load_custom_multihead_crf_model(model_path: str, device: str = "auto"):
    """
    Utility function to load a Multi-Head CRF model.

    Args:
        model_path: Path to the saved model directory
        device: Device to load model on ("auto", "cpu", "cuda", etc.)

    Returns:
        tuple: (model, tokenizer, config)
    """
    from transformers import AutoTokenizer
    import os

    # Validate model directory
    required_files = ["config.json", "modeling.py"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]

    if missing_files:
        raise FileNotFoundError(f"Missing required files in {model_path}: {missing_files}")

    print(f"Loading Multi-Head CRF model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load config
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config_dict = json.load(f)

    config = MultiHeadCRFConfig(**config_dict)

    # Load model
    model = TokenClassificationModelMultiHeadCRF.from_pretrained(model_path, config=config)

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model type: {type(model).__name__}")
    print(f"Entity types: {model.entity_types}")
    print(f"Number of labels per head: {model.num_labels}")

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


# Register the MultiHeadCRF config for auto loading
try:
    from transformers import AutoConfig
    AutoConfig.register("multihead-crf-tagger", MultiHeadCRFConfig)
except Exception:
    pass  # Config may already be registered
