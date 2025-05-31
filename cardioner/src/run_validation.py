'''
This is meant as a validation script to apply trained models via the huggingface
transformer pipeline. The --input_folder is a directory of jsonl's with {'id': .. , 'text': "bla", 'tags': [{'start': xx, 'end': xy, 'tag': "bla"}]}
'''

import argparse
from transformers import pipeline
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict, Literal
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.nl import Dutch
from spacy.lang.it import Italian
from spacy.lang.ro import Romanian
from spacy.lang.sv import Swedish
from spacy.lang.cs import Czech
from tqdm import tqdm

from utils import process_pipe, merge_annotations
from predictor import PredictionNER

lang_dict = {
    'es': Spanish,
    'nl': Dutch,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'sv': Swedish,
    'cz': Czech
}

def main(model: str, revision: str, lang: str, ignore_zero: bool, input_dir :str,
         stride: int, batchwise: bool, batch_size: int, annotation_tsv: str, file_prefix: str,
         split_by_class: bool=False, custom_predictor: bool=False,
         confidence_threshold: float=0.35, **kwargs):

    sample_list = merge_annotations(annotation_directory=input_dir, annotation_tsv=annotation_tsv)

    print(f"There are {len(sample_list)} validation samples")
    res_df_raw = pd.DataFrame()
    if not custom_predictor:
        tokenizer = AutoTokenizer.from_pretrained(model,
            truncation=True, padding='max_length', model_max_length=512, padding_side='right', truncation_side='right')
        le_pipe = pipeline('token-classification',
                            model=model,
                            revision=revision,
                            tokenizer=tokenizer,
                            aggregation_strategy="simple",
                            batch_size=16 if batchwise else None,
                            device=0)

        if batchwise==False:
            for sample in tqdm(sample_list):
                named_ents = process_pipe(text=sample['text'], lang=lang, pipe = le_pipe, max_word_per_chunk=stride, hf_stride=True)
                if len(named_ents)>0:
                    _res_df = pd.DataFrame(named_ents)
                    _res_df['id'] = sample['id']
                    if ignore_zero:
                        _res_df = _res_df[_res_df.entity_group!='LABEL_0']
                    res_df_raw = pd.concat([res_df_raw, _res_df], axis=0)
        else:
            print("Performing inference in batch mode")
            from datasets import Dataset

            # Create a dataset from the sample_list
            ner_dataset = Dataset.from_dict({
                'text': [sample['text'] for sample in sample_list],
                'id': [sample['id'] for sample in sample_list]
            })
            results = process_pipe(text=ner_dataset, lang=lang, pipe = le_pipe, max_word_per_chunk=stride, hf_stride=True, batch_size=batch_size)
            print("finished..")
            # Each result in results corresponds to one sample in the dataset/sample_list
            for i, doc_results in enumerate(results):
                if len(doc_results) > 0:
                    _res_df = pd.DataFrame(doc_results)
                    _res_df['id'] = sample_list[i]['id']  # Get ID from the original sample_list
                    if ignore_zero:
                        _res_df = _res_df[_res_df.entity_group != 'LABEL_0']
                    res_df_raw = pd.concat([res_df_raw, _res_df], axis=0)
    else:
        ner_pipe = PredictionNER(model_checkpoint=model, revision=revision)
        res_list = []
        for sample in tqdm(sample_list):
            res = ner_pipe.do_prediction(sample['text'], confidence_threshold=confidence_threshold)
            if len(res)>0:
                for _res in res:
                    res_list.append({
                        'id': sample['id'],
                        'entity_group': _res['tag'],
                        'ann_id': 'NA',
                        'start': _res['start'],
                        'end': _res['end'],
                        'word': _res['text']
                    })
        res_df_raw = pd.DataFrame(res_list)

    res_df_raw = res_df_raw.rename(columns={'start': 'start_span', 'end': 'end_span', 'entity_group': 'label', 'word': 'text', 'id': 'filename'})
    res_df_raw['ann_id'] = "NAN"
    res_df_raw['filename'] = res_df_raw['filename'].str.strip()
    if split_by_class:
        labels = res_df_raw.label.unique().tolist()
        for lab in labels:
            res_df_raw.loc[res_df_raw.label==lab,
                ['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']]\
            .to_csv(f'{file_prefix}results_{lab}.tsv', sep="\t", index=False)

    else:
        res_df_raw[['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']].to_csv(f'{file_prefix}results.tsv', sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained model, input a directory with jsonls, outputs a tsv with results')
    parser.add_argument('--model', type=str, help='The model to test, can be a path or a model name', default='StivenLancheros/mBERT-base-Biomedical-NER')
    parser.add_argument('--revision', type=str, help='Model revision, optional', default=None)
    parser.add_argument('--lang', type=str, help='The language of the text', choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'], required=True)
    parser.add_argument('--ignore_zero', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--split_by_class', action='store_true')
    parser.add_argument('--file_prefix', type=str, default="")
    parser.add_argument('--batchwise', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--custom_predictor', action='store_true')
    parser.add_argument('--confidence_threshold', type=float, default=0.35)
    parser.add_argument('--annotation_tsv', type=str, help='Annotation file, only for folder with txts', default=None)

    args = parser.parse_args()
    main(**vars(args))
