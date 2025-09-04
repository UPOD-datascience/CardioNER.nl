# Custom Multi-Label Token Classification Model

This document explains how to use and load custom multi-label token classification models that have been created with custom classifier hidden layers.

## Overview

When you create a model with `classifier_hidden_layers` as a tuple (e.g., `(512, 256)`), the system creates a custom model class with a multi-layer perceptron (MLP) classifier head instead of the standard single linear layer. This custom model provides more flexibility and potentially better performance but requires special handling for loading.

## Model Architecture

### Standard Model (classifier_hidden_layers=None)
```
Input → RoBERTa → Dropout → Linear(hidden_size, num_labels) → Output
```

### Custom Model (classifier_hidden_layers=(512, 256))
```
Input → RoBERTa → Dropout → Linear(hidden_size, 512) → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout → Linear(256, num_labels) → Output
```

## Creating a Custom Model

To create a custom model, simply pass a tuple of hidden layer sizes to the `classifier_hidden_layers` parameter:

```python
from cardioner.multilabel.trainer import ModelTrainer

trainer = ModelTrainer(
    label2id=your_label2id,
    id2label=your_id2label,
    classifier_hidden_layers=(512, 256),  # This creates a custom model
    classifier_dropout=0.1,
    freeze_backbone=True,
    # ... other parameters
)

# Train and save the model
trainer.train(train_data, test_data, eval_data)
```

## What Happens During Saving

When you save a custom model, the system automatically:

1. **Copies the model definition**: The `modeling.py` file containing `MultiLabelTokenClassificationModelCustom` is copied to your output directory.

2. **Updates the config**: The model configuration is updated with:
   - `auto_map`: Points to the custom model class
   - `classifier_hidden_layers`: Stores the layer configuration
   - `classifier_dropout`: Stores the dropout rate
   - `freeze_backbone`: Stores the backbone freezing setting

3. **Saves all components**: Model weights, tokenizer, and configuration are saved together.

## Loading a Custom Model

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

### Method 2: Direct Import (for local use)

```python
from cardioner.multilabel.modeling import MultiLabelTokenClassificationModelCustom
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/your/model")
model = MultiLabelTokenClassificationModelCustom.from_pretrained("path/to/your/model")
```

## Important Notes

### Security Consideration
- `trust_remote_code=True` allows execution of custom code from the model directory
- Only use this with models from trusted sources
- The custom model code is saved in `modeling.py` in your model directory

### Model Configuration
The following custom parameters are automatically saved in the config:
- `freeze_backbone`: Whether the backbone was frozen during training
- `classifier_hidden_layers`: Tuple defining the MLP architecture
- `classifier_dropout`: Dropout rate used in the classifier

### Compatibility
- Custom models are **not directly compatible** with standard HuggingFace Hub uploads
- They require the `modeling.py` file and `trust_remote_code=True` to load
- For Hub compatibility, use `classifier_hidden_layers=None` (standard model)

## Example Usage

See `load_custom_model_example.py` for a complete example of loading and using a custom model:

```python
# Example inference
inputs = tokenizer("Patient has chest pain.", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities > 0.5).int()
```

## Troubleshooting

### "No module named 'modeling'" Error
- Ensure `modeling.py` exists in your model directory
- Check that you're using `trust_remote_code=True`

### Model Loading Fails
- Verify the model was saved with `classifier_hidden_layers` as a tuple
- Check that all required files are present: `config.json`, `pytorch_model.bin`, `modeling.py`

### Performance Issues
- Custom models with many hidden layers may be slower
- Consider the trade-off between model complexity and inference speed
- Monitor GPU memory usage with larger classifier heads

## File Structure

After saving a custom model, your output directory should contain:

```
your_model_directory/
├── config.json              # Model configuration with auto_map
├── pytorch_model.bin         # Model weights
├── tokenizer_config.json     # Tokenizer configuration
├── tokenizer.json           # Tokenizer vocabulary
├── modeling.py              # Custom model class definition
└── training_args.json       # Training arguments (optional)
```

## Migration from Standard Model

If you have a standard model and want to convert it to use custom layers:

1. Retrain with `classifier_hidden_layers` as a tuple
2. The model architecture will change, so you cannot directly transfer weights
3. Consider the performance implications before switching

## Best Practices

1. **Start Simple**: Begin with `classifier_hidden_layers=None` and only add complexity if needed
2. **Monitor Performance**: Track both accuracy and inference speed
3. **Document Configuration**: Keep track of the hidden layer configurations that work best
4. **Version Control**: Save different model configurations for comparison
5. **Security**: Only load custom models from trusted sources
