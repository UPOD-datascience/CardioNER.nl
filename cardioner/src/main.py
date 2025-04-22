from os import environ
import os
import spacy

# Load a spaCy model for tokenization
nlp = spacy.blank("nl")
environ["WANDB_MODE"] = "disabled"
environ["WANDB_DISABLED"] = "true"
environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import List, Dict, Optional, Literal
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
import argparse
from functools import partial

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.utils import shuffle

from multilabel.trainer import ModelTrainer as MultiLabelModelTrainer
from multiclass.trainer import ModelTrainer as MultiClassModelTrainer

import evaluate
metric = evaluate.load("seqeval")



# https://huggingface.co/learn/nlp-course/en/chapter7/2

# TODO: add paragraph splitter from Lorenzo
# TODO: add per label performance

def prepare(Model: str='CLTL/MedRoBERTa.nl',
         lang: str='en',
         corpus_train: Optional[str]=None,
         corpus_validation: Optional[str]=None,
         split_file: Optional[str]=None,
         annotation_loc: Optional[str] = None,
         label2id: Optional[Dict[str, int]]=None,
         id2label: Optional[Dict[int, str]]=None,
         chunk_size: int=256,
         max_length: int=514,
         chunk_type: Literal['standard', 'centered', 'paragraph']='standard',
         multi_class: bool=False,
         use_iob: bool=True,
         hf_token: str=None
         ):

    if multi_class:
        from multiclass.loader import annotate_corpus_standard
        from multiclass.loader import annotate_corpus_centered
        from multiclass.loader import tokenize_and_align_labels
    else:
        from multilabel.loader import annotate_corpus_paragraph
        from multilabel.loader import annotate_corpus_standard
        from multilabel.loader import annotate_corpus_centered
        from multilabel.loader import tokenize_and_align_labels
        from multilabel.loader import count_tokens_with_multiple_labels

    corpus_train_list = []
    # read jsonl
    with open(corpus_train, 'r', encoding='utf-8') as fr:
        for line in fr:
            corpus_train_list.append(json.loads(line))

    corpus_validation_list = []
    if corpus_validation is not None:
        # read jsonl
        with open(corpus_validation, 'r', encoding='utf-8') as fr:
            for line in fr:
                corpus_validation_list.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(Model, add_prefix_space=True, token=hf_token)
    tokenizer.model_max_length = max_length

    model_config = AutoConfig.from_pretrained(Model, token=hf_token)
    max_model_length = model_config.max_position_embeddings  # 514
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)  # 2
    max_allowed_chunk_size = max_model_length - num_special_tokens  # 512

    if split_file is not None:
        with open(split_file, 'r', encoding='utf-8') as fr:
            split_data = json.load(fr)
            # TODO: check if the split file is correct, seems redundant to have separate entries for class/language.
            corpus_train_ids = split_data[lang]['train']['symp']
            corpus_test_ids = split_data[lang]['test']['symp']
            corpus_validation_ids = split_data[lang]['validation']['symp']

            print(f"{len(corpus_train_ids)} training samples, {len(corpus_test_ids)} test samples, {len(corpus_validation_ids)} validation samples")

        corpus_train = [entry for entry in corpus_train_list if entry['id'] in corpus_train_ids]
        corpus_test = [entry for entry in corpus_train_list if entry['id'] in corpus_test_ids]
        corpus_validation = [entry for entry in corpus_train_list if entry['id'] in corpus_validation_ids]
    else:
        corpus_train = corpus_train_list
        corpus_validation = corpus_validation_list
        corpus_test = []

    # Run the transformation
    annotate_functions = {
        'standard': annotate_corpus_standard,
        'centered': annotate_corpus_centered,
        'paragraph': annotate_corpus_paragraph
    }

    annotate_kwargs = {
        'standard': {
            'chunk_size': chunk_size,
            'max_allowed_chunk_size': max_allowed_chunk_size,
            'IOB': use_iob
        },
        'paragraph': {
            'chunk_size': chunk_size,
            'max_allowed_chunk_size': max_allowed_chunk_size,
            'IOB': use_iob
        },
        'centered': {
            'chunk_size': chunk_size,
            'IOB': use_iob
        }
    }

    datasets = {
        'train': corpus_train,
        'test': corpus_test,
        'validation': corpus_validation
    }

    # Remove datasets that are None
    datasets = {k: v for k, v in datasets.items() if v is not None}

    # Initialize variables
    iob_data_train = iob_data_test = iob_data_validation = []
    unique_tags = None

    annotate_func = annotate_functions[chunk_type]
    kwargs = annotate_kwargs[chunk_type]

    for batch_id, corpus in datasets.items():
        iob_data, _unique_tags = annotate_func(corpus, batch_id=batch_id, **kwargs)

        if batch_id == 'train':
            iob_data_train = iob_data
            unique_tags = _unique_tags
        elif batch_id == 'test':
            iob_data_test = iob_data
        elif batch_id == 'validation':
            iob_data_validation = iob_data

        if (batch_id != 'train') & (len(corpus)>0):
            assert(unique_tags == _unique_tags), "Tags are not the same in train/val"
    # paragraph?

    label2id = {l:int(c) for c,l in enumerate(unique_tags)}
    id2label = {int(c):l for c,l in enumerate(unique_tags)}

    print("Unique tags: ", unique_tags)

    iob_data = iob_data_train + iob_data_test + iob_data_validation

    if multi_class==False:
        count_tokens_with_multiple_labels(iob_data)


    partial_tokenize_and_align_labels = partial(tokenize_and_align_labels,
                                                tokenizer=tokenizer,
                                                label2id=label2id,
                                                max_length=max_length)

    iob_data_dataset = Dataset.from_list(iob_data)
    iob_data_dataset_tokenized = iob_data_dataset.map(
        partial_tokenize_and_align_labels,
        batched=True,
    )

    # given a max_length tokens we want to center the context around all spans in the documents and extract them as a seperate documents. Each separate extraction needs to get a separate sub_id.

    max_seq_length = max(len(entry['input_ids']) for entry in iob_data_dataset_tokenized)
    print(f"Maximum sequence length after tokenization: {max_seq_length}")

    iob_data_dataset_tokenized_with_labels = []
    for entry in iob_data_dataset_tokenized:
        entry.update({'label2id': label2id, 'id2label': id2label})
        iob_data_dataset_tokenized_with_labels.append(entry)

    if annotation_loc is not None:
        annotation_loc = annotation_loc.replace('.jsonl', f'_chunk{ChunkSize}_{ChunkType}.jsonl')
        with open(annotation_loc, 'w', encoding='utf-8') as fw:
            for entry in iob_data_dataset_tokenized_with_labels:
                json.dump(entry, fw)
                fw.write('\n')

    return iob_data_dataset_tokenized_with_labels, unique_tags

def train(tokenized_data_train: List[Dict],
          tokenized_data_test: List[Dict],
          tokenized_data_validation: List[Dict],
          force_splitter: bool=False,
          Model: str='CLTL/MedRoBERTa.nl',
          Splits: List[List[str]] | int = 5,
          output_dir: str="../output",
          max_length: int=514,
          num_epochs: int=10,
          batch_size: int=20,
          profile: bool=False,
          multi_class: bool=False,
          use_crf: bool=False,
          weight_decay: float=0.001,
          learning_rate: float=1e-4,
          accumulation_steps: int=1,
          hf_token: str=None,
          freeze_backbone: bool=False,
          classifier_hidden_layers: tuple|None=None,
          classifier_dropout: float=0.1,

    label2id = tokenized_data_train[0]['label2id']
    id2label = tokenized_data_train[0]['id2label']

    label2id_tr = tokenized_data_train[0]['label2id']
    if (tokenized_data_validation is not None) & (len(tokenized_data_validation)>0):
        label2id_vl = tokenized_data_validation[0]['label2id']
        assert(label2id_tr==label2id_vl), "Label2id mismatch between train, validation."

    if (tokenized_data_test is not None) & (len(tokenized_data_test)>0):
        label2d_test = tokenized_data_train[0]['label2id']
        assert(label2id_tr==label2d_test), "Label2id mismatch between train, test."

    label2id = {str(k):int(v) for k,v in label2id.items()}
    id2label = {int(k):str(v) for k,v in id2label.items()}

    # Ensure labels are correct
    num_labels = len(label2id)
    for entry in tokenized_data_train:
        labels = entry['labels']
        if multi_class == False:
            for token_labels in labels:
                if isinstance(token_labels, list):
                    assert len(token_labels) == num_labels, "Mismatch in label dimensions, in train set."
                else:
                    assert token_labels == -100, "Labels should be lists or -100."
        else:
            assert all([label in [-100]+list(range(num_labels)) for label in labels]), f"Labels should be in range (0,{num_labels})  or -100."

    for entry in tokenized_data_validation:
        labels = entry['labels']
        if multi_class == False:
            for token_labels in labels:
                if isinstance(token_labels, list):
                    assert len(token_labels) == num_labels, "Mismatch in label dimensions, in validation set."
                else:
                    assert token_labels == -100, "Labels should be lists or -100."
        else:
            assert all([label in [-100]+list(range(num_labels)) for label in labels]), f"Labels should be in range (0,{num_labels})  or -100."

    if (tokenized_data_validation is None) | ((tokenized_data_validation is not None) & (force_splitter==True)):
        print("Using cross-validation for model training and validation..")
        if isinstance(Splits, int):
            splitter = GroupKFold(n_splits=Splits)
            groups = [entry['gid'] for entry in tokenized_data_train]
            shuffled_data, shuffled_groups = shuffle(tokenized_data_train, groups, random_state=42)
            SplitList = list(splitter.split(shuffled_data, groups=shuffled_groups))
        else:
            SplitList = Splits

        print(f"Splitting data into {len(SplitList)} folds")
        for k, (train_idx, test_idx) in enumerate(SplitList):
            if multi_class:
                TrainClass = MultiClassModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None,
                                    model=Model, output_dir=f"{output_dir}/fold_{k}",
                                    max_length=max_length,
                                    num_train_epochs=num_epochs,
                                    batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    learning_rate=learning_rate,
                                    gradient_accumulation_steps=accumulation_steps,
                                    hf_token=hf_token)
            else:
                TrainClass = MultiLabelModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None,
                                    model=Model, output_dir=f"{output_dir}/fold_{k}",
                                    max_length=max_length,
                                    num_train_epochs=num_epochs,
                                    batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    learning_rate=learning_rate,
                                    gradient_accumulation_steps=accumulation_steps,
                                    hf_token=hf_token,
                                    freeze_backbone=freeze_backbone)

            print(f"Training on split {k}")
            train_data = [shuffled_data[i] for i in train_idx]
            test_data = [shuffled_data[i] for i in test_idx]

            if (tokenized_data_validation is not None) & (force_splitter==True):
                TrainClass.train(train_data=train_data, test_data=test_data, eval_data=test_data, profile=profile)
            else:
                TrainClass.train(train_data=train_data, test_data=[], eval_data=tokenized_data_validation, profile=profile)
    elif (tokenized_data_validation is not None) and (tokenized_data_test is not None):
        print("Using preset train/test/validation split for model training and validation..")
        if multi_class:
            TrainClass = MultiClassModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None,
                                model=Model, use_crf=use_crf, output_dir=output_dir,
                                max_length=max_length,
                                num_train_epochs=num_epochs,
                                batch_size=batch_size,
                                weight_decay=weight_decay,
                                learning_rate=learning_rate,
                                gradient_accumulation_steps=accumulation_steps,
                                hf_token=hf_token)
        else:
            TrainClass = MultiLabelModelTrainer(label2id=label2id, id2label=id2label, tokenizer=None,
                                model=Model, output_dir=output_dir,
                                max_length=max_length,
                                num_train_epochs=num_epochs,
                                batch_size=batch_size,
                                weight_decay=weight_decay,
                                learning_rate=learning_rate,
                                gradient_accumulation_steps=accumulation_steps,
                                hf_token=hf_token,
                                freeze_backbone=freeze_backbone)

        print("Training on full dataset")
        TrainClass.train(train_data=tokenized_data_train, test_data=tokenized_data_test, eval_data=tokenized_data_validation, profile=profile)
    else:
        raise ValueError("No validation data provided, and no cross-validation splits provided. Please provide either a validation set or a split file.")

if __name__ == "__main__":
    """
        take in .jsonl with:
        {'tags': [{'start': int, 'end': int, 'tag': str}], 'text':str, 'id': str}

        and output .jsonl with tokenized and aligned data
    """
    #TODO: add gradient accumulation
    argparsers = argparse.ArgumentParser()
    argparsers.add_argument('--model', type=str, default='CLTL/MedRoBERTa.nl')
    argparsers.add_argument('--lang', type=str, required=True, choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'])
    argparsers.add_argument('--corpus_train', type=str, required=False)
    argparsers.add_argument('--corpus_validation', type=str, required=False)
    argparsers.add_argument('--split_file', type=str, required=False)
    argparsers.add_argument('--annotation_loc', type=str, required=False)
    argparsers.add_argument('--output_dir', type=str, default="../../output")
    argparsers.add_argument('--parse_annotations', action="store_true", default=False)
    argparsers.add_argument('--train_model', action="store_true", default=False)
    argparsers.add_argument("--freeze_backbone", action="store_true", help="Freeze the transformer backbone and train only the classifier head")
    argparsers.add_argument('--chunk_size', type=int, default=None)
    argparsers.add_argument('--chunk_type', type=str, default='standard', choices=['standard', 'centered', 'paragraph'])
    argparsers.add_argument('--max_token_length', type=int, default=514)
    argparsers.add_argument('--num_epochs', type=int, default=10)
    argparsers.add_argument('--num_labels', type=int, default=9)
    argparsers.add_argument('--learning_rate', type=float, default=1e-4)
    argparsers.add_argument('--weight_decay', type=float, default=0.01)
    argparsers.add_argument('--batch_size', type=int, default=16)
    argparsers.add_argument('--accumulation_steps', type=int, default=1)
    argparsers.add_argument('--num_splits', type=int, default=5)
    argparsers.add_argument('--hf_token', type=str, default=None)
    argparsers.add_argument('--multiclass', action="store_true", default=False)
    argparsers.add_argument('--use_crf', action="store_true", default=False)
    argparsers.add_argument('--profile', action="store_true", default=False)
    argparsers.add_argument('--force_splitter', action="store_true", default=False)
    argparsers.add_argument('--write_annotations', action="store_true", default=False)
    argparsers.add_argument('--without_iob_tagging', action="store_true", default=False)
    argparsers.add_argument('--classifier_hidden_layers', type=int, nargs='+', default=None)
    argparsers.add_argument('--classifier_dropout', type=float, default=0.1)


    args = argparsers.parse_args()

    tokenized_data = None
    tags = None

    _model = args.model
    corpus_train = args.corpus_train
    corpus_validation = args.corpus_validation
    split_file = args.split_file
    force_splitter =args.force_splitter
    _annotation_loc = args.annotation_loc
    parse_annotations = args.parse_annotations
    train_model = args.train_model
    hf_token = args.hf_token
    freeze_backbone = args.freeze_backbone
    classifier_hidden_layers = args.classifier_hidden_layers
    classifier_dropout = args.classifier_dropout

    if args.without_iob_tagging:
        use_iob = False
        print("WARNING: you are training without the IOB-tagging scheme. Ensure this is correct.")
    else:
        use_iob = True

    assert(((corpus_train is not None) and (corpus_validation is not None)) | ((split_file is not None) and (corpus_train is not None)) | ((corpus_train is not None) and (force_splitter))), "Either provide a split file or a train and validation corpus"
    assert((_annotation_loc is not None) | (parse_annotations is not None)), "Either provide an annotation location or set parse_annotations to True"
    assert((train_model is True) | (parse_annotations is True)), "Either parse annotations or train the model, or both..do something!"

    if corpus_train is not None:
        assert os.path.isfile(corpus_train), f"Corpus_train file {corpus_train} does not exist."
    if corpus_validation is not None:
        assert os.path.isfile(corpus_validation), f"Corpus_validation file {corpus_validation} does not exist."
    if split_file is not None:
        assert os.path.isfile(split_file), f"Split_file {split_file} does not exist."

    ################################################
    ################################################

    if (split_file is not None) and (corpus_validation is not None):
        print("Split file and validation corpus provided, ignoring validation corpus file in favor of split file.")
        corpus_validation = None

    if force_splitter:
        if corpus_validation is not None:
            print("Force splitter is on, and validation set is provided, therefore, ignoring split file.")
            split_file = None
        elif split_file is not None:
            print("Force splitter is on, and split file is provided, therefore, ignoring the test in the split file (if present).")

    OutputDir = args.output_dir
    ChunkSize = args.max_token_length if args.chunk_size is None else args.chunk_size
    max_length = args.max_token_length
    ChunkType = args.chunk_type
    num_labels = args.num_labels
    num_splits = args.num_splits
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    profile = args.profile
    multi_class = args.multiclass
    use_crf = args.use_crf
    language = args.lang
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate
    accumulation_steps = args.accumulation_steps

    if args.write_annotations == False:
        _annotation_loc = None

    if parse_annotations:
        print("Loading and prepping data..")
        tokenized_data, tags = prepare(Model=_model,
                                    lang = language,
                                    corpus_train=corpus_train,
                                    corpus_validation = corpus_validation,
                                    split_file=split_file,
                                    annotation_loc=_annotation_loc,
                                    chunk_size=ChunkSize,
                                    chunk_type=ChunkType,
                                    max_length=max_length,
                                    multi_class = multi_class,
                                    use_iob=use_iob,
                                    hf_token=hf_token)

    if train_model:
        print("Training the model..")
        if tokenized_data is None:
            with open(_annotation_loc, 'r', encoding='utf-8') as fr:
                tokenized_data = [json.loads(line) for line in fr]

        # check if input_ids and labels have the same length, are smaller than max_length and if the labels are within range
        for entry in tokenized_data:
            assert(len(entry['input_ids']) == len(entry['labels'])), f"Input_ids and labels have different lengths for entry {entry['id']}"
            assert(len(entry['input_ids']) <= max_length), f"Input_ids are longer than max_length for entry {entry['id']}"

            # assert that all labels are within range, >=0 and < num_labels
            for label in entry['labels']:
                if multi_class == False:
                    assert(all([((_label >= 0) and (_label < num_labels)) | (_label==-100) for _label in label])), f"Label {label}, {type(label)} is not within range for entry {entry['id']}"
                else:
                    assert(label in [-100]+list(range(num_labels))), f"Label {label} is not within range for entry {entry['id']}"

        # TODO: this is extremely ugly and needs to be refactored
        tokenized_data_train = [entry for entry in tokenized_data if entry['batch'] == 'train']
        tokenized_data_test = [entry for entry in tokenized_data if entry['batch'] == 'test']
        tokenized_data_validation = [entry for entry in tokenized_data if entry['batch'] == 'validation']

        if len(tokenized_data_train)==0:
            tokenized_data_train = tokenized_data

        if (len(tokenized_data_validation)==0) and (len(tokenized_data_test)>0):
            tokenized_data_validation = tokenized_data_test

        # train-> split-> validation (if train is available, validation is available, and force_splitter = True)
        # train-> test -> validation (if train is available, test is available, and validation is available)
        # [x] train-> validation (if train is available, and validation is available)
        train(tokenized_data_train,
              tokenized_data_test,
              tokenized_data_validation,
              force_splitter=force_splitter,
              Model=_model,
              Splits=num_splits,
              output_dir=OutputDir,
              max_length=max_length,
              num_epochs=num_epochs,
              batch_size=batch_size,
              profile=profile,
              multi_class=multi_class,
              use_crf=use_crf,
              weight_decay=weight_decay,
              learning_rate=learning_rate,
              accumulation_steps=accumulation_steps,
              hf_token=hf_token,
              freeze_backbone=freeze_backbone,
              classifier_hidden_layers=classifier_hidden_layers,
              classifier_dropout=classifier_dropout)
