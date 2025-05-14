'''
This is meant as a validation script to apply trained models via the huggingface
transformer pipeline. The --input_folder is a directory of jsonl's with {'id': .. , 'text': "bla", 'tags': [{'start': xx, 'end': xy, 'tag': "bla"}]}
'''

import argparse
from transformers import pipeline
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

lang_dict = {
    'es': Spanish,
    'nl': Dutch,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'sv': Swedish,
    'cz': Czech
}

def main(model, lang, ignore_zero, input_dir):
    le_pipe = pipeline('ner',
                        model=model,
                        tokenizer=model,
                        aggregation_strategy="simple",
                        device=-1)

    sample_list = merge_annotations(annotation_directory=input_dir)

    print(f"There are {len(sample_list)} validation samples")
    res_df_raw = pd.DataFrame()
    for sample in tqdm(sample_list):
        named_ents = process_pipe(text=sample['text'], pipe = le_pipe)
        if len(named_ents)>0:
            _res_df = pd.DataFrame(named_ents)
            _res_df['id'] = sample['id']
            if ignore_zero:
                _res_df = _res_df[_res_df.entity_group!='LABEL_0']
            res_df_raw = pd.concat([res_df_raw, _res_df], axis=0)

    res_df_raw = res_df_raw.rename(columns={'start': 'start_span', 'end': 'end_span', 'entity_group': 'label', 'word': 'text', 'id': 'filename'})
    res_df_raw['ann_id'] = "NAN"
    res_df_raw = res_df_raw.sort_values(by=['filename', 'start_span'])
    res_df_raw['filename'] = res_df_raw['filename'].str.strip()
    res_df_raw[['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']].to_csv('results.tsv', sep="\t", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained model, input a directory with jsonls, outputs a tsv with results')
    parser.add_argument('--model', type=str, help='The model to test, can be a path or a model name', default='StivenLancheros/mBERT-base-Biomedical-NER')
    parser.add_argument('--lang', type=str, help='The language of the text', choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'], required=True)
    parser.add_argument('--ignore_zero', action='store_true', default=False)
    parser.add_argument('--input_dir', type=str, required=True)

    args = parser.parse_args()

    main(**vars(args))
