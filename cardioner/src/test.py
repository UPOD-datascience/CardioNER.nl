'''
This is meant as a test script to apply trained models via the huggingface
transformer pipeline.
'''

import argparse
from transformers import pipeline
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

lang_dict = {
    'es': Spanish,
    'nl': Dutch,
    'en': English,
    'it': Italian,
    'ro': Romanian,
    'sv': Swedish,
    'cz': Czech
}

def process_pipe(text: str,
                 pipe: pipeline,
                 max_word_per_chunk: int=256,
                 lang: Literal['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz']='en') -> List[Dict[str, str]]:
    '''
      text: The text to process
      pipe: The transformers pipeline to use
      max_word_per_chunk: The maximum number of words per chunk, we need this to avoid exceeding the maximum input size of the model
      lang: The language of the text
    '''
    assert(lang in ['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz']), f"Language {lang} not supported"

    nlp = lang_dict[lang]()
    nlp.add_pipe('sentencizer')

    doc = nlp(text)

    sentence_bag = []
    word_count = 0
    named_ents = []
    for sent in tqdm(doc.sents):
        word_count += len(sent)
        if word_count > max_word_per_chunk:
            _named_ents = pipe(".".join(sentence_bag))
            named_ents.extend(_named_ents)
            sentence_bag = []
            word_count = len(sent)
        sentence_bag.append(sent.text)
    if len(sentence_bag) > 0:
        _named_ents = pipe(".".join(sentence_bag))
        named_ents.extend(_named_ents)

    return named_ents

def main(model, lang):
    le_pipe = pipeline('ner', 
                        model=model, 
                        tokenizer=model, aggregation_strategy="simple", 
                        device=-1)

    named_ents = process_pipe(text=text_dict[lang], pipe = le_pipe)

    print("*"*50)
    print(pd.DataFrame(named_ents))
    print("*"*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--model', type=str, help='The model to test, can be a path or a model name', default='StivenLancheros/mBERT-base-Biomedical-NER')
    parser.add_argument('--lang', type=str, help='The language of the text', choices=['es', 'nl', 'en', 'it', 'ro', 'sv', 'cz'])
    args = parser.parse_args()

    main(args.model, args.lang)

