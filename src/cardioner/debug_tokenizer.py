from transformers import AutoTokenizer

def test_tokenizer_behavior(model_name, test_text="This is a test sentence with medical terms."):
    print(f"Testing tokenizer for model: {model_name}")
    print(f"Test text: '{test_text}'")
    print("="*60)

    # Load tokenizer with different configurations
    print("1. Testing with default settings:")
    tokenizer_default = AutoTokenizer.from_pretrained(model_name)
    tokens_default = tokenizer_default.tokenize(test_text)
    print(f"   Tokens: {tokens_default}")

    print("\n2. Testing with truncation=False, padding=None, use_fast=True:")
    tokenizer_custom = AutoTokenizer.from_pretrained(model_name, truncation=False, padding=None, use_fast=True)
    tokens_custom = tokenizer_custom.tokenize(test_text)
    print(f"   Tokens: {tokens_custom}")

    print("\n3. Testing encoding with return_offsets_mapping:")
    encoded = tokenizer_custom(test_text, return_offsets_mapping=True, return_tensors="pt")
    print(f"   Input IDs: {encoded['input_ids']}")
    print(f"   Offset mapping: {encoded.get('offset_mapping', 'Not available')}")

    print("\n4. Tokenizer properties:")
    print(f"   Tokenizer class: {type(tokenizer_custom)}")
    print(f"   Is fast: {tokenizer_custom.is_fast}")
    print(f"   Model max length: {tokenizer_custom.model_max_length}")

    if hasattr(tokenizer_custom, '_tokenizer'):
        print(f"   Has _tokenizer: True")
        if hasattr(tokenizer_custom._tokenizer, 'model'):
            print(f"   Has _tokenizer.model: True")
            prefix = getattr(tokenizer_custom._tokenizer.model, 'continuing_subword_prefix', None)
            print(f"   Original continuing_subword_prefix: {repr(prefix)}")
        else:
            print(f"   Has _tokenizer.model: False")
    else:
        print(f"   Has _tokenizer: False")

if __name__ == "__main__":
    # Test with your model
    model_name = "/media/bramiozo/Storage2/DATA/NER/DT4H_results/NL/CardioBerta_clinical/multilabel_iob_test"  # Replace with your actual model
    test_tokenizer_behavior(model_name)
