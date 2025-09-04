#!/usr/bin/env python3
"""
Example script demonstrating how to load a custom MultiLabel Token Classification model
that was saved with custom classifier hidden layers.

This model requires trust_remote_code=True to load because it uses a custom model class.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

# Import utility functions from modeling.py
try:
    from modeling import load_custom_cardioner_model, validate_custom_model_directory
except ImportError:
    print("Warning: Could not import utility functions from modeling.py")
    print("Using fallback implementation...")

    def load_custom_cardioner_model(model_path: str, device: str = "auto"):
        """Fallback implementation"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, tokenizer, model.config

    def validate_custom_model_directory(model_path: str):
        """Fallback validation"""
        required_files = ["config.json", "modeling.py", "pytorch_model.bin"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        return {"valid": len(missing_files) == 0, "errors": missing_files}

def load_custom_model(model_path: str):
    """
    Load a custom MultiLabel Token Classification model with trust_remote_code=True

    This is a wrapper around the utility function for backward compatibility.

    Args:
        model_path: Path to the saved model directory

    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer, config = load_custom_cardioner_model(model_path)

    print(f"Model type: {type(model).__name__}")
    print(f"Number of labels: {model.num_labels}")
    print(f"Label mapping: {model.config.id2label}")

    # Print custom configuration if available
    if hasattr(model.config, 'classifier_hidden_layers'):
        print(f"Custom classifier layers: {model.config.classifier_hidden_layers}")
    if hasattr(model.config, 'freeze_backbone'):
        print(f"Backbone frozen: {model.config.freeze_backbone}")
    if hasattr(model.config, 'classifier_dropout'):
        print(f"Classifier dropout: {model.config.classifier_dropout}")

    return model, tokenizer

def test_inference(model, tokenizer, text: str):
    """
    Test inference with the loaded model

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input text for testing
    """
    print(f"\nTesting inference with text: '{text}'")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply sigmoid for multi-label probabilities
        probabilities = torch.sigmoid(logits)

        # Get predictions (threshold at 0.5)
        predictions = (probabilities > 0.5).int()

    print(f"Logits shape: {logits.shape}")
    print(f"Max probability: {probabilities.max().item():.4f}")
    print(f"Min probability: {probabilities.min().item():.4f}")
    print(f"Number of predicted labels: {predictions.sum().item()}")

    # Print top predictions per token
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
        if token in ['<s>', '</s>', '<pad>']:
            continue
        token_probs = probabilities[0, i]
        top_labels = torch.topk(token_probs, k=min(3, len(token_probs)))

        print(f"Token '{token}':")
        for j, (prob, label_idx) in enumerate(zip(top_labels.values, top_labels.indices)):
            label = model.config.id2label[label_idx.item()]
            print(f"  {j+1}. {label}: {prob.item():.4f}")
        print()

def validate_and_load_model(model_path: str):
    """
    Validate model directory and load the model with proper error handling

    Args:
        model_path: Path to the saved model directory

    Returns:
        tuple: (model, tokenizer) if successful, None otherwise
    """
    print(f"Validating model directory: {model_path}")

    # First, validate the model directory
    validation_results = validate_custom_model_directory(model_path)

    if not validation_results["valid"]:
        print("❌ Model validation failed!")
        print("Errors:")
        for error in validation_results["errors"]:
            print(f"  - {error}")
        return None, None

    if validation_results["warnings"]:
        print("⚠️  Validation warnings:")
        for warning in validation_results["warnings"]:
            print(f"  - {warning}")

    print("✅ Model validation passed!")
    print("Files found:")
    for file_info in validation_results["files_found"]:
        print(f"  - {file_info}")

    if "model_info" in validation_results and validation_results["model_info"]:
        print("\nModel Information:")
        for key, value in validation_results["model_info"].items():
            print(f"  - {key}: {value}")

    try:
        # Load the model using utility function
        print(f"\nLoading model...")
        model, tokenizer, config = load_custom_cardioner_model(model_path)
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def main():
    """
    Main function demonstrating model validation, loading and inference
    """
    # Example usage - replace with your actual model path
    model_path = "path/to/your/saved/model"

    print("Custom CardioNER Model Loading Example")
    print("=" * 40)

    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print("\nPlease update the model_path variable with the correct path to your saved model.")
        print("\nExample paths:")
        print("  - './output/cardioner_model'")
        print("  - '/path/to/your/model/directory'")
        return

    try:
        # Validate and load the custom model
        model, tokenizer = validate_and_load_model(model_path)

        if model is None:
            print("\n❌ Failed to load model. Please check the errors above.")
            return

        print(f"\n✅ Model loaded successfully!")

        # Test with example text
        test_text = "Patient has chest pain and shortness of breath."
        test_inference(model, tokenizer, test_text)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model was saved with custom classifier_hidden_layers (tuple)")
        print("2. Check that the modeling.py file exists in the model directory")
        print("3. Verify you have the correct transformers version installed")
        print("4. Make sure all required files are present in the model directory")

if __name__ == "__main__":
    main()
