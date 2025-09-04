#!/usr/bin/env python3
"""
Example script demonstrating how to load a custom multiclass Token Classification model
that was saved with custom classifier configurations or CRF layers.

This model requires trust_remote_code=True to load because it uses custom model classes.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

# Import utility functions from modeling.py
try:
    from modeling import load_custom_cardioner_multiclass_model, validate_custom_multiclass_model_directory
except ImportError:
    print("Warning: Could not import utility functions from modeling.py")
    print("Using fallback implementation...")

    def load_custom_cardioner_multiclass_model(model_path: str, device: str = "auto"):
        """Fallback implementation"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, tokenizer, model.config

    def validate_custom_multiclass_model_directory(model_path: str):
        """Fallback validation"""
        required_files = ["config.json", "modeling.py", "pytorch_model.bin"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        return {"valid": len(missing_files) == 0, "errors": missing_files}

def load_custom_model(model_path: str):
    """
    Load a custom multiclass Token Classification model with trust_remote_code=True

    This is a wrapper around the utility function for backward compatibility.

    Args:
        model_path: Path to the saved model directory

    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer, config = load_custom_cardioner_multiclass_model(model_path)

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

    # Check if CRF is used
    model_class_name = type(model).__name__
    if "CRF" in model_class_name or hasattr(model, 'crf'):
        print(f"Model uses CRF layer: Yes")
    else:
        print(f"Model uses CRF layer: No")

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

        # For multiclass, get predictions using argmax
        if hasattr(model, 'crf'):
            # For CRF models, use Viterbi decoding
            try:
                predictions = model.crf.decode(logits, mask=inputs['attention_mask'].bool())
                predictions = torch.tensor(predictions[0])  # Take first batch item
            except:
                # Fallback to simple argmax if CRF decode fails
                predictions = torch.argmax(logits, dim=-1)[0]
        else:
            # For regular models, use argmax
            predictions = torch.argmax(logits, dim=-1)[0]

        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)

    print(f"Logits shape: {logits.shape}")
    print(f"Max probability: {probabilities.max().item():.4f}")
    print(f"Min probability: {probabilities.min().item():.4f}")

    # Print token-wise predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"\nToken-wise predictions:")
    print(f"{'Token':<15} {'Predicted Label':<20} {'Confidence':<10}")
    print("-" * 45)

    for i, (token, pred_idx) in enumerate(zip(tokens, predictions)):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        pred_label = model.config.id2label.get(pred_idx.item(), f"ID_{pred_idx.item()}")
        confidence = probabilities[0, i, pred_idx].item()

        print(f"{token:<15} {pred_label:<20} {confidence:<10.4f}")

def test_crf_specific_features(model, tokenizer, text: str):
    """
    Test CRF-specific features if the model has CRF

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        text: Input text for testing
    """
    if not hasattr(model, 'crf'):
        print("Model does not have CRF layer - skipping CRF-specific tests")
        return

    print(f"\nTesting CRF-specific features with text: '{text}'")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Get best sequence using Viterbi decoding
        best_sequence = model.crf.decode(logits, mask=inputs['attention_mask'].bool())

        # Get log partition function
        try:
            log_likelihood = model.crf(logits, inputs['attention_mask'].bool())
            print(f"Log-likelihood of best sequence: {log_likelihood.item():.4f}")
        except:
            print("Could not compute log-likelihood")

        print(f"Best sequence (CRF decoded): {best_sequence[0]}")

        # Compare with simple argmax
        argmax_sequence = torch.argmax(logits, dim=-1)[0].tolist()
        print(f"Argmax sequence: {argmax_sequence}")

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
    validation_results = validate_custom_multiclass_model_directory(model_path)

    if not validation_results["valid"]:
        print("❌ Model validation failed!")
        print("Errors:")
        for error in validation_results["errors"]:
            print(f"  - {error}")
        return None, None

    if validation_results.get("warnings", []):
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
        model, tokenizer, config = load_custom_cardioner_multiclass_model(model_path)
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

    print("Custom CardioNER Multiclass Model Loading Example")
    print("=" * 50)

    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print("\nPlease update the model_path variable with the correct path to your saved model.")
        print("\nExample paths:")
        print("  - './output/cardioner_multiclass_model'")
        print("  - '/path/to/your/multiclass/model/directory'")
        return

    try:
        # Validate and load the custom model
        model, tokenizer = validate_and_load_model(model_path)

        if model is None:
            print("\n❌ Failed to load model. Please check the errors above.")
            return

        print(f"\n✅ Model loaded successfully!")

        # Test with example texts
        test_texts = [
            "Patient has chest pain and shortness of breath.",
            "The ECG shows ST elevation in leads V2-V4.",
            "Blood pressure is 140/90 mmHg with tachycardia."
        ]

        for i, test_text in enumerate(test_texts, 1):
            print(f"\n{'='*20} Test {i} {'='*20}")
            test_inference(model, tokenizer, test_text)

            # Test CRF features if available
            if hasattr(model, 'crf'):
                test_crf_specific_features(model, tokenizer, test_text)

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model was saved with the updated trainer")
        print("2. Check that the modeling.py file exists in the model directory")
        print("3. Verify you have the required dependencies installed:")
        print("   - torch")
        print("   - transformers")
        print("   - torchcrf (for CRF models)")
        print("4. Make sure all required files are present in the model directory")
        print("5. Try loading with a simple script first:")
        print("   model = AutoModelForTokenClassification.from_pretrained(path, trust_remote_code=True)")

if __name__ == "__main__":
    main()
