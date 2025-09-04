#!/usr/bin/env python3
"""
Comprehensive demonstration script for loading custom CardioNER models.

This script shows how to load both multilabel and multiclass models that have been
saved with custom configurations and require trust_remote_code=True.

Usage:
    python load_custom_models_demo.py --multilabel_model /path/to/multilabel/model --multiclass_model /path/to/multiclass/model
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Add the cardioner modules to path for imports
sys.path.append(str(Path(__file__).parent))

def load_multilabel_model(model_path: str) -> Tuple[Any, Any, Dict]:
    """
    Load a custom multilabel CardioNER model.

    Args:
        model_path: Path to the saved multilabel model

    Returns:
        tuple: (model, tokenizer, config)
    """
    print(f"Loading multilabel model from: {model_path}")

    # Try to use utility function first
    try:
        from multilabel.modeling import load_custom_cardioner_model
        return load_custom_cardioner_model(model_path)
    except ImportError:
        print("Using fallback loading method for multilabel model...")

    # Fallback method
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, model.config

def load_multiclass_model(model_path: str) -> Tuple[Any, Any, Dict]:
    """
    Load a custom multiclass CardioNER model.

    Args:
        model_path: Path to the saved multiclass model

    Returns:
        tuple: (model, tokenizer, config)
    """
    print(f"Loading multiclass model from: {model_path}")

    # Try to use utility function first
    try:
        from multiclass.modeling import load_custom_cardioner_multiclass_model
        return load_custom_cardioner_multiclass_model(model_path)
    except ImportError:
        print("Using fallback loading method for multiclass model...")

    # Fallback method
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer, model.config

def validate_model_directory(model_path: str, model_type: str) -> bool:
    """
    Validate that a model directory contains required files.

    Args:
        model_path: Path to model directory
        model_type: Type of model ('multilabel' or 'multiclass')

    Returns:
        bool: True if valid, False otherwise
    """
    required_files = ["config.json", "pytorch_model.bin", "modeling.py"]

    print(f"Validating {model_type} model directory: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model directory does not exist: {model_path}")
        return False

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print(f"✅ {model_type.capitalize()} model directory validation passed")
    return True

def print_model_info(model: Any, tokenizer: Any, config: Dict, model_type: str):
    """
    Print information about the loaded model.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Model configuration
        model_type: Type of model ('multilabel' or 'multiclass')
    """
    print(f"\n{model_type.upper()} MODEL INFORMATION")
    print("=" * 40)
    print(f"Model class: {type(model).__name__}")
    print(f"Number of labels: {getattr(model, 'num_labels', 'Unknown')}")
    print(f"Device: {next(model.parameters()).device}")

    # Print label mappings (limited to avoid clutter)
    if hasattr(config, 'id2label') and config.id2label:
        print(f"Labels (showing first 10): {dict(list(config.id2label.items())[:10])}")
        if len(config.id2label) > 10:
            print(f"... and {len(config.id2label) - 10} more labels")

    # Print custom configuration
    custom_attrs = [
        'classifier_hidden_layers',
        'classifier_dropout',
        'freeze_backbone',
        'custom_model_type'
    ]

    for attr in custom_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            print(f"{attr}: {value}")

    # Check for CRF (multiclass only)
    if model_type == 'multiclass' and hasattr(model, 'crf'):
        print("CRF layer: Yes")
        print(f"CRF num_tags: {model.crf.num_tags}")
    elif model_type == 'multiclass':
        print("CRF layer: No")

def test_multilabel_inference(model: Any, tokenizer: Any, text: str):
    """
    Test inference with a multilabel model.

    Args:
        model: Multilabel model
        tokenizer: Tokenizer
        text: Test text
    """
    print(f"\nTesting multilabel inference with: '{text}'")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply sigmoid for multilabel probabilities
        probabilities = torch.sigmoid(logits)

        # Get predictions (threshold at 0.5)
        predictions = (probabilities > 0.5).int()

    print(f"Logits shape: {logits.shape}")
    print(f"Max probability: {probabilities.max().item():.4f}")
    print(f"Min probability: {probabilities.min().item():.4f}")
    print(f"Total predicted labels: {predictions.sum().item()}")

    # Show top predictions for first few tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("\nTop predictions (first 5 tokens):")
    for i, token in enumerate(tokens[:5]):
        if token in ['<s>', '</s>', '<pad>']:
            continue
        token_probs = probabilities[0, i]
        top_k = min(3, len(token_probs))
        top_indices = torch.topk(token_probs, k=top_k).indices
        print(f"'{token}':")
        for idx in top_indices:
            label = model.config.id2label.get(idx.item(), f"ID_{idx.item()}")
            prob = token_probs[idx].item()
            print(f"  {label}: {prob:.4f}")

def test_multiclass_inference(model: Any, tokenizer: Any, text: str):
    """
    Test inference with a multiclass model.

    Args:
        model: Multiclass model
        tokenizer: Tokenizer
        text: Test text
    """
    print(f"\nTesting multiclass inference with: '{text}'")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Handle CRF vs regular models
        if hasattr(model, 'crf'):
            # CRF model - use Viterbi decoding
            try:
                predictions = model.crf.decode(logits, mask=inputs['attention_mask'].bool())
                predictions = torch.tensor(predictions[0])  # Take first batch
                print("Used CRF Viterbi decoding")
            except Exception as e:
                print(f"CRF decoding failed ({e}), using argmax fallback")
                predictions = torch.argmax(logits, dim=-1)[0]
        else:
            # Regular model - use argmax
            predictions = torch.argmax(logits, dim=-1)[0]

        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)

    print(f"Logits shape: {logits.shape}")
    print(f"Max probability: {probabilities.max().item():.4f}")
    print(f"Min probability: {probabilities.min().item():.4f}")

    # Show token-wise predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print("\nToken-wise predictions (first 10 tokens):")
    print(f"{'Token':<15} {'Predicted':<20} {'Confidence':<10}")
    print("-" * 45)

    for i, (token, pred_idx) in enumerate(zip(tokens[:10], predictions[:10])):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        pred_label = model.config.id2label.get(pred_idx.item(), f"ID_{pred_idx.item()}")
        confidence = probabilities[0, i, pred_idx].item()

        print(f"{token:<15} {pred_label:<20} {confidence:<10.4f}")

def run_demo(multilabel_path: Optional[str] = None, multiclass_path: Optional[str] = None):
    """
    Run the complete demonstration.

    Args:
        multilabel_path: Path to multilabel model (optional)
        multiclass_path: Path to multiclass model (optional)
    """
    print("CardioNER Custom Models Loading Demo")
    print("=" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    # Test sentences
    test_sentences = [
        "Patient has chest pain and shortness of breath.",
        "ECG shows ST elevation in leads V2-V4 with Q waves.",
        "Blood pressure 140/90 mmHg, heart rate 102 bpm, irregular rhythm."
    ]

    # Test multilabel model if path provided
    if multilabel_path:
        print("MULTILABEL MODEL TESTING")
        print("=" * 50)

        if validate_model_directory(multilabel_path, "multilabel"):
            try:
                ml_model, ml_tokenizer, ml_config = load_multilabel_model(multilabel_path)
                print_model_info(ml_model, ml_tokenizer, ml_config, "multilabel")

                # Test inference
                for i, sentence in enumerate(test_sentences[:2]):  # Test first 2 sentences
                    print(f"\n--- Test {i+1} ---")
                    test_multilabel_inference(ml_model, ml_tokenizer, sentence)

            except Exception as e:
                print(f"❌ Error loading multilabel model: {e}")
        else:
            print("Skipping multilabel model due to validation failures.")
    else:
        print("No multilabel model path provided - skipping multilabel tests")

    print("\n" + "="*70 + "\n")

    # Test multiclass model if path provided
    if multiclass_path:
        print("MULTICLASS MODEL TESTING")
        print("=" * 50)

        if validate_model_directory(multiclass_path, "multiclass"):
            try:
                mc_model, mc_tokenizer, mc_config = load_multiclass_model(multiclass_path)
                print_model_info(mc_model, mc_tokenizer, mc_config, "multiclass")

                # Test inference
                for i, sentence in enumerate(test_sentences[:2]):  # Test first 2 sentences
                    print(f"\n--- Test {i+1} ---")
                    test_multiclass_inference(mc_model, mc_tokenizer, sentence)

            except Exception as e:
                print(f"❌ Error loading multiclass model: {e}")
        else:
            print("Skipping multiclass model due to validation failures.")
    else:
        print("No multiclass model path provided - skipping multiclass tests")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Demo script for loading custom CardioNER models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test both model types
    python load_custom_models_demo.py --multilabel_model ./models/multilabel --multiclass_model ./models/multiclass

    # Test only multilabel
    python load_custom_models_demo.py --multilabel_model ./models/multilabel

    # Test only multiclass
    python load_custom_models_demo.py --multiclass_model ./models/multiclass

    # Interactive mode (will prompt for paths)
    python load_custom_models_demo.py
        """
    )

    parser.add_argument(
        "--multilabel_model",
        type=str,
        help="Path to the multilabel model directory"
    )

    parser.add_argument(
        "--multiclass_model",
        type=str,
        help="Path to the multiclass model directory"
    )

    args = parser.parse_args()

    # Interactive mode if no arguments provided
    if not args.multilabel_model and not args.multiclass_model:
        print("No model paths provided. Enter paths manually (or press Enter to skip):")

        multilabel_input = input("Multilabel model path: ").strip()
        args.multilabel_model = multilabel_input if multilabel_input else None

        multiclass_input = input("Multiclass model path: ").strip()
        args.multiclass_model = multiclass_input if multiclass_input else None

        if not args.multilabel_model and not args.multiclass_model:
            print("No model paths provided. Exiting.")
            return

    try:
        run_demo(args.multilabel_model, args.multiclass_model)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*40)
    print("Demo completed!")
    print("\nTips:")
    print("- Always use trust_remote_code=True when loading custom models")
    print("- Check that modeling.py exists in your model directories")
    print("- Install torchcrf for CRF models: pip install torchcrf")
    print("- Validate model directories before loading")

if __name__ == "__main__":
    main()
