'''
This is meant as a test script to apply trained models via the huggingface
transformer pipeline.
'''

import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from typing import List, Dict, Literal
from example_texts import text_dict
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.nl import Dutch
from spacy.lang.it import Italian
from spacy.lang.ro import Romanian
from spacy.lang.sv import Swedish
from spacy.lang.cs import Czech
from tqdm import tqdm
from torch.cuda import is_available
from torch import bfloat16
from utils import process_pipe

lang_dict = {
    'es': Spanish,
    'nl': Dutch,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'sv': Swedish,
    'cz': Czech
}

def main(model_name, lang, ignore_zero, stride):
    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=bfloat16
    )
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    le_pipe = pipeline('ner',
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple",
                        device=0 if is_available() else -1,
                        trust_remote_code=True)

    named_ents = process_pipe(text=text_dict[lang], pipe = le_pipe, max_word_per_chunk=stride)
    res_df = pd.DataFrame(named_ents)

    if ignore_zero:
        res_df = res_df[res_df.entity_group!='LABEL_0']

    print("*"*50)
    print(res_df)
    print("*"*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--model_name', type=str, help='The model to test, can be a path or a model name', default='StivenLancheros/mBERT-base-Biomedical-NER')
    parser.add_argument('--lang', type=str, help='The language of the text', choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'])
    parser.add_argument('--ignore_zero', action='store_true', default=False)
    parser.add_argument('--stride', type=int, help='Stride of the NER inference')

    args = parser.parse_args()

    main(**vars(args))
