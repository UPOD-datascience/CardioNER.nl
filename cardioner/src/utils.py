import pprint
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List
import hashlib
import os
import json

def pretty_print_classifier(classifier):
    """Pretty print a PyTorch module with indentation and structure details."""
    import pprint
    import torch.nn as nn

    pp = pprint.PrettyPrinter(indent=4, width=100)

    if isinstance(classifier, nn.Linear):
        return f"Linear(in_features={classifier.in_features}, out_features={classifier.out_features})"

    if isinstance(classifier, nn.Sequential):
        # Create a structured representation as a dictionary
        layers_info = {}
        for i, layer in enumerate(classifier):
            layer_info = {
                'type': layer.__class__.__name__
            }

            # Add details based on layer type
            if isinstance(layer, nn.Linear):
                layer_info['in_features'] = layer.in_features
                layer_info['out_features'] = layer.out_features
            elif isinstance(layer, nn.Dropout):
                layer_info['p'] = layer.p
            elif isinstance(layer, nn.BatchNorm1d):
                layer_info['num_features'] = layer.num_features
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                layer_info['in_channels'] = layer.in_channels
                layer_info['out_channels'] = layer.out_channels

            layers_info[f'layer_{i}'] = layer_info

        # Use pprint to format the dictionary
        return f"Sequential with {len(classifier)} layers:\n{pp.pformat(layers_info)}"

    # For other module types, convert to dict and pretty print
    try:
        # Try to extract meaningful attributes
        module_info = {
            'type': classifier.__class__.__name__,
            'parameters': {name: param.size() for name, param in classifier.named_parameters()}
        }
        return pp.pformat(module_info)
    except:
        # Fallback
        return str(classifier)

def calculate_class_weights(dataset, label2id, multiclass=False, smoothing_factor=0.05):
    """
    Calculate class weights based on label frequency in the dataset.

    Args:
        dataset: Training dataset containing token labels
        label2id: Dictionary mapping label names to ids
        multilabel: Whether labels are multilabel (one-hot encoded)
        smoothing_factor: Smoothing factor to avoid extreme weights

    Returns:
        List of class weights
    """
    label_counts = defaultdict(int)
    total_tokens = 0

    print("Extracting class weights from training set...")
    # Count label occurrences
    for example in tqdm(dataset):
        labels = example['labels']

        if multiclass==False:
            # For multilabel: labels are one-hot encoded [seq_length, num_labels]
            # Convert to tensor if it's a list
            labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels

            # Skip padding tokens (assuming padding tokens have all zeros or special value)
            valid_tokens = (labels_tensor.sum(dim=-1) > 0) if labels_tensor.ndim > 1 else (labels_tensor != -100)

            # Count each class occurrence
            if labels_tensor.ndim > 1:  # One-hot encoded
                for i in range(labels_tensor.shape[1]):  # For each class
                    class_occurrences = labels_tensor[:, i].sum().item()
                    label_counts[i] += class_occurrences
                    total_tokens += valid_tokens.sum().item()
            else:
                # Handle case where labels might be indices instead of one-hot
                for label_idx in labels_tensor[valid_tokens]:
                    label_counts[label_idx.item()] += 1
                    total_tokens += 1
        else:
            # For single-label classification
            for label in labels:
                if label != -100:  # Skip padding tokens
                    label_counts[label] += 1
                    total_tokens += 1

    # Calculate weights inversely proportional to frequency
    weights = []
    num_classes = len(label2id)

    for i in range(num_classes):
        count = label_counts.get(i, 0)
        # Apply smoothing to avoid division by zero and extreme values
        weight = (total_tokens / (count + smoothing_factor * total_tokens))
        weights.append(weight)

    # Normalize weights to have an average of 1
    weights = [w / sum(weights) * len(weights) for w in weights]

    id2label = {i: label for i, label in enumerate(label2id)}
    print(f'Extracted class weights: {[f"{id2label[i]}_{str(w)}" for i,w in enumerate(weights)]}')
    return weights

def merge_annotations(annotation_directory: str, merge_key: str='id', tag_key: str='tags', text_key: str='text')->List[Dict]:
    # go through .jsonl's in directory
    #
    annotations = []
    file_processed = False
    for _file in tqdm(os.listdir(annotation_directory)):
        if _file.endswith('.jsonl'):
            file = os.path.join(annotation_directory, _file)
            with open(file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    annotations.append(json.loads(line))
            file_processed = True
    if not file_processed:
        raise ValueError("No JSONL file found in the directory, maybe change the extension?")

    NEW_DICT = defaultdict(lambda : defaultdict(list))
    for d in tqdm(annotations):
        # list of tags
        tags = d[tag_key]
        text = d[text_key]
        id = d[merge_key]

        assert(tags is not None), f"Tags should not be none: {id}"

        NEW_DICT[id][tag_key].extend(tags)
        NEW_DICT[id][text_key].extend([text])

    NEW_DICT_LIST = []
    for k,v in tqdm(NEW_DICT.items()):
        # check if text is consistent
        #
        list_of_texts = v[text_key]
        set_of_hashes = set()
        for t in list_of_texts:
            _hash = hashlib.md5(t.encode('utf-8')).hexdigest()
            set_of_hashes.add(_hash)

        if len(set_of_hashes)>1:
            print(f"Skipping {k} because there are {len(set_of_hashes)} different texts")
            continue

        _d = {
            'id': k,
            'tags': v[tag_key],
            'text': list_of_texts[0]
        }
        NEW_DICT_LIST.append(_d)
    return NEW_DICT_LIST
