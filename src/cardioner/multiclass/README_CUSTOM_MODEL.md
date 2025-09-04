# Custom Multiclass Token Classification Model

This document explains how to use and load custom multiclass token classification models that have been created with custom classifier configurations, CRF layers, or other custom features.

## Overview

The multiclass CardioNER system supports two types of custom models:

1. **TokenClassificationModel**: Standard token classification with configurable MLP classifier heads
2. **TokenClassificationModelCRF**: Token classification with Conditional Random Fields (CRF) for sequence modeling

Both model types require `trust_remote_code=True` to load because they use custom model classes that extend the standard HuggingFace implementations.

## Model Architecture

### Standard Model (use_crf=False)
```
Input → RoBERTa → Dropout → Classifier Head → CrossEntropy Loss → Output
```

### CRF Model (use_crf=True)
```
Input → RoBERTa → Dropout → Classifier Head → CRF Layer → CRF Loss → Output
```

### Configurable Classifier Heads

Both models support configurable classifier heads:

**Simple Head (classifier_hidden_layers=None):**
```
hidden_size → Linear(hidden_size, num_labels)
```

**Custom MLP Head (classifier_hidden_layers=(512, 256)):**
```
hidden_size → Linear(512) → ReLU → Dropout → Linear(256) → ReLU → Dropout → Linear(num_labels)
```

## Creating Custom Models

### Standard Multiclass Model
```python
from cardioner.multiclass.trainer import ModelTrainer

trainer = ModelTrainer(
    label2id=your_label2id,
    id2label=your_id2label,
    use_crf=False,  # Standard model
    classifier_hidden_layers=(512, 256),  # Optional custom head
    classifier_dropout=0.1,
    freeze_backbone=False,
    # ... other parameters
)

# Train and save the model
trainer.train(train_data, test_data, eval_data)
```

### CRF-Enhanced Model
```python
from cardioner.multiclass.trainer import ModelTrainer

trainer = ModelTrainer(
    label2id=your_label2id,
    id2label=your_id2label,
    use_crf=True,  # Enable CRF
    classifier_hidden_layers=(512, 256),  # Optional custom head
    classifier_dropout=0.1,
    freeze_backbone=False,
    # ... other parameters
)

# Train and save the model
trainer.train(train_data, test_data, eval_data)
```

## What Happens During Saving

When you save a custom multiclass model, the system automatically:

1. **Copies the model definition**: The `modeling.py` file containing both `TokenClassificationModel` and `TokenClassificationModelCRF` is copied to your output directory.

2. **Updates the config**: The model configuration is updated with:
   - `auto_map`: Points to the appropriate custom model class
   - `classifier_hidden_layers`: Stores the layer configuration
   - `classifier_dropout`: Stores the dropout rate
   - `class_weights`: Stores class weights if provided
   - `custom_model_type`: Identifies the model type
   - `requires_trust_remote_code`: Flags that custom code is needed

3. **Saves all components**: Model weights, tokenizer, and configuration are saved together.

## Loading Custom Models

### Method 1: Using AutoModelForTokenClassification (Recommended)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")

# Load model with trust_remote_code=True
model = AutoModelForTokenClassification.from_pretrained(
    "path/to/your/model",
    trust_remote_code=True  # Required for custom model classes!
)
```

### Method 2: Using Utility Functions

```python
from cardioner.multiclass.modeling import load_custom_cardioner_multiclass_model

model, tokenizer, config = load_custom_cardioner_multiclass_model("path/to/your/model")
```

### Method 3: Direct Import (for local use)

```python
from cardioner.multiclass.modeling import TokenClassificationModel, TokenClassificationModelCRF
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")

# For CRF models
model = TokenClassificationModelCRF.from_pretrained("path/to/your/model")

# For standard models
model = TokenClassificationModel.from_pretrained("path/to/your/model")
```

## Model-Specific Features

### CRF Models

CRF models provide additional functionality:

#### Viterbi Decoding
```python
# Get best sequence using CRF
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    best_sequence = model.crf.decode(logits, mask=inputs['attention_mask'].bool())
```

#### Log-Likelihood Calculation
```python
# Calculate sequence log-likelihood
log_likelihood = model.crf(logits, attention_mask.bool())
```

#### Structured Prediction
CRF models enforce valid label sequences and can capture label dependencies better than standard models.

### Standard Models

Standard models use simple token-wise classification:

```python
# Get predictions using argmax
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

## Important Notes

### Security Consideration
- `trust_remote_code=True` allows execution of custom code from the model directory
- Only use this with models from trusted sources
- The custom model code is saved in `modeling.py` in your model directory

### Dependencies

For CRF models, you need:
```bash
pip install torchcrf
```

### Model Configuration

The following custom parameters are automatically saved in the config:
- `use_crf`: Whether CRF is enabled (inferred from model class)
- `freeze_backbone`: Whether the backbone was frozen during training
- `classifier_hidden_layers`: Tuple defining the MLP architecture
- `classifier_dropout`: Dropout rate used in the classifier
- `class_weights`: Class weights for handling imbalanced data

### Compatibility
- Custom models are **not directly compatible** with standard HuggingFace Hub uploads
- They require the `modeling.py` file and `trust_remote_code=True` to load
- For Hub compatibility, avoid custom configurations

## Example Usage

See `load_custom_model_example.py` for complete examples:

```python
# Example inference
inputs = tokenizer("Patient has chest pain.", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    
    if hasattr(model, 'crf'):
        # CRF model - use Viterbi decoding
        predictions = model.crf.decode(outputs.logits, mask=inputs['attention_mask'].bool())
    else:
        # Standard model - use argmax
        predictions = torch.argmax(outputs.logits, dim=-1)
```

## Troubleshooting

### "No module named 'modeling'" Error
- Ensure `modeling.py` exists in your model directory
- Check that you're using `trust_remote_code=True`

### "torchcrf not found" Error
- Install torchcrf: `pip install torchcrf`
- This is required for CRF models

### Model Loading Fails
- Verify the model was saved with the updated trainer
- Check that all required files are present: `config.json`, `pytorch_model.bin`, `modeling.py`
- Ensure you have the correct dependencies installed

### CRF Decoding Issues
- Check that attention masks are provided
- Verify that the CRF layer was properly initialized
- Make sure labels are in the correct range for CRF

### Performance Issues
- CRF models are slower than standard models
- Custom classifier heads with many layers increase computation time
- Monitor GPU memory usage with complex architectures

## File Structure

After saving a custom model, your output directory should contain:

```
your_model_directory/
├── config.json              # Model configuration with auto_map
├── pytorch_model.bin         # Model weights
├── tokenizer_config.json     # Tokenizer configuration
├── tokenizer.json           # Tokenizer vocabulary
├── modeling.py              # Custom model class definitions
└── training_args.json       # Training arguments (optional)
```

## Model Comparison

| Feature | Standard Model | CRF Model |
|---------|---------------|-----------|
| Speed | Fast | Slower |
| Memory | Low | Higher |
| Sequence Modeling | Token-wise | Structured |
| Label Dependencies | No | Yes |
| Inference Method | Argmax | Viterbi |
| Training Stability | Stable | May need tuning |

## Best Practices

1. **Choose the Right Model**:
   - Use CRF for tasks requiring structured output (e.g., BIO tagging)
   - Use standard models for simpler classification tasks

2. **Start Simple**: Begin with standard models and simple classifier heads

3. **Monitor Performance**: Track both accuracy and inference speed

4. **Validate Thoroughly**: Test both types of inference (argmax vs CRF decode)

5. **Handle Dependencies**: Ensure torchcrf is available in deployment environments

6. **Version Control**: Save different model configurations for comparison

7. **Security**: Only load custom models from trusted sources

## Migration Guide

### From Standard to CRF Model
1. Retrain with `use_crf=True`
2. Model architecture changes significantly - cannot transfer weights directly
3. Adjust inference code to use CRF decoding
4. Install torchcrf dependency

### From Simple to Custom Classifier Head
1. Retrain with `classifier_hidden_layers` parameter
2. Model head architecture changes - cannot transfer head weights
3. Monitor for potential overfitting with complex heads

## Advanced Usage

### Custom Loss Functions
Both models support custom loss functions through subclassing:

```python
class CustomTokenClassificationModel(TokenClassificationModel):
    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        # Add custom loss computation
        return outputs
```

### Multi-GPU Training
Both models support standard PyTorch multi-GPU training:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Fine-tuning Strategies
- **Freeze backbone**: Set `freeze_backbone=True` for head-only training
- **Gradual unfreezing**: Unfreeze layers progressively during training
- **Layer-wise learning rates**: Use different learning rates for different components