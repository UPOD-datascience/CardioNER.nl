import pprint
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Literal, Optional
import hashlib
import os
import json
import pandas as pd
from transformers import pipeline

from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.nl import Dutch
from spacy.lang.it import Italian
from spacy.lang.ro import Romanian
from spacy.lang.sv import Swedish
from spacy.lang.cs import Czech

lang_dict = {
    'es': Spanish,
    'nl': Dutch,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'sv': Swedish,
    'cz': Czech
}

def _offset_tags(tags, offset):
    replacement = []
    for _tags in tags:
        _tags['start'] += offset
        _tags['end'] += offset
        replacement.append(_tags)
    return replacement

def process_pipe(text: str,
                 pipe: pipeline,
                 max_word_per_chunk: int=256,
                 hf_stride: bool=True,
                 lang: Literal['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz']='en') -> List[Dict[str, str]]:
    '''
      text: The text to process
      pipe: The transformers pipeline to use
      max_word_per_chunk: The maximum number of words per chunk,
        we need this to avoid exceeding the maximum input size of the model
      lang: The language of the text
    '''
    assert(lang in ['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz']), f"Language {lang} not supported"

    nlp = lang_dict[lang]()
    nlp.add_pipe('sentencizer')
    doc = nlp(text)

    if hf_stride:
        named_ents = pipe(text, stride=max_word_per_chunk)
    else:
        # only necessary if no FastTokenizer is available or aggregation_strategy is None
        sentence_bag = []
        word_count = 0
        named_ents = []
        offset = 0
        for sent in doc.sents:
            word_count += len(sent)
            if word_count > max_word_per_chunk:
                txt = " ".join(sentence_bag)
                _named_ents = pipe(txt)
                # add offsets
                if offset>0:
                    _named_ents =_offset_tags(_named_ents, offset)
                named_ents.extend(_named_ents)
                sentence_bag = []
                word_count = len(sent)
                offset += len(txt)
            sentence_bag.append(sent.text)
        if len(sentence_bag) > 0:
            _named_ents = pipe(".".join(sentence_bag))
            named_ents.extend(_named_ents)
            if offset>0:
                _named_ents = _offset_tags(_named_ents, offset)
    return named_ents

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

def merge_annotations(annotation_directory: str, merge_key: str='id', tag_key: str='tags', text_key: str='text', annotation_tsv: str|None=None)->List[Dict]:
    # go through .jsonl's/.txts in directory
    #

    if isinstance(annotation_tsv, str):
        annotation_df = pd.read_csv(annotation_tsv, sep='\t')
    else:
        annotation_df = None
    annotations = []
    for _file in tqdm(os.listdir(annotation_directory)):
        if _file.endswith('.jsonl'):
            file = os.path.join(annotation_directory, _file)
            with open(file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    annotations.append(json.loads(line))
        elif _file.endswith('.txt'):
            if annotation_df is None:
                raise ValueError("The annotation tsv needs to be provided if you parse a list of .txts")
            file = os.path.join(annotation_directory, _file)
            with open(file, 'r', encoding='utf-') as fr:
                text = fr.read()
            fn = _file.split('.')[0]
            d = {merge_key: fn, text_key: text}
            r = annotation_df.query(f'filename=="{fn.strip()}"')[['label', 'start_span', 'end_span']]
            d[tag_key] = [{'tag': row[0], 'start': row[1], 'end': row[2] } for row in r]
            annotations.append(d)
    if len(annotations)==0:
        raise ValueError("No JSONL or TXT file found in the directory, maybe check the directory?")

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
