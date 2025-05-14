mod_info_dict = {}
mod_info_dict['sapbert'] ='''
For more details about training and eval, see SapBERT [github repo](https://github.com/cambridgeltl/sapbert).


### Citation
```bibtex
@inproceedings{liu-etal-2021-self,
    title = "Self-Alignment Pretraining for Biomedical Entity Representations",
    author = "Liu, Fangyu  and
      Shareghi, Ehsan  and
      Meng, Zaiqiao  and
      Basaldella, Marco  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.334",
    pages = "4228--4238",
    abstract = "Despite the widespread success of self-supervised learning via masked language models (MLM), accurately capturing fine-grained semantic relationships in the biomedical domain remains a challenge. This is of paramount importance for entity-level tasks such as entity linking where the ability to model entity relations (especially synonymy) is pivotal. To address this challenge, we propose SapBERT, a pretraining scheme that self-aligns the representation space of biomedical entities. We design a scalable metric learning framework that can leverage UMLS, a massive collection of biomedical ontologies with 4M+ concepts. In contrast with previous pipeline-based hybrid systems, SapBERT offers an elegant one-model-for-all solution to the problem of medical entity linking (MEL), achieving a new state-of-the-art (SOTA) on six MEL benchmarking datasets. In the scientific domain, we achieve SOTA even without task-specific supervision. With substantial improvement over various domain-specific pretrained MLMs such as BioBERT, SciBERTand and PubMedBERT, our pretraining scheme proves to be both effective and robust.",
}
```
For more details about training/eval and other scripts, see CardioNER [github repo](https://github.com/DataTools4Heart/CardioNER).
and for more information on the background, see Datatools4Heart [Huggingface](https://huggingface.co/DT4H)/[Website](https://www.datatools4heart.eu/)

'''

mod_info_dict['mirrorbert'] ='''
For more details about training and eval, see MirrorBERT [github repo](https://github.com/cambridgeltl/mirror-bert).


### Citation
```bibtex
@inproceedings{liu-etal-2021-fast,
    title = "Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders",
    author = "Liu, Fangyu  and
      Vuli{\'c}, Ivan  and
      Korhonen, Anna  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.109",
    pages = "1442--1459",
}
```
For more details about training/eval and other scripts, see CardioNER [github repo](https://github.com/DataTools4Heart/CardioNER).
and for more information on the background, see Datatools4Heart [Huggingface](https://huggingface.co/DT4H)/[Website](https://www.datatools4heart.eu/)

'''


mod_info_dict['cardioner'] ='''
For more details about training/eval and other scripts, see CardioNER [github repo](https://github.com/DataTools4Heart/CardioNER).
and for more information on the background, see Datatools4Heart [Huggingface](https://huggingface.co/DT4H)/[Website](https://www.datatools4heart.eu/)
'''


# https://huggingface.co/docs/hub/repositories-licenses
licenses = [
            'apache-2.0', 'mit', 'openrail', 'bigscience-openrail-m', 'creativeml-openrail-m',
            'bigscience-bloom-rail-1.0', 'bigcode-openrail-m', 'afl-3.0', 'artistic-2.0', 'bsl-1.0',
            'bsd', 'bsd-2-clause', 'bsd-3-clause', 'bsd-3-clause-clear', 'c-uda', 'cc', 'cc0-1.0',
            'cc-by-2.0', 'cc-by-2.5', 'cc-by-3.0', 'cc-by-4.0', 'cc-by-sa-3.0', 'cc-by-sa-4.0',
            'cc-by-nc-2.0', 'cc-by-nc-3.0', 'cc-by-nc-4.0', 'cc-by-nd-4.0', 'cc-by-nc-nd-3.0',
            'cc-by-nc-nd-4.0', 'cc-by-nc-sa-2.0', 'cc-by-nc-sa-3.0', 'cc-by-nc-sa-4.0',
            'cdla-sharing-1.0', 'cdla-permissive-1.0', 'cdla-permissive-2.0', 'wtfpl', 'ecl-2.0',
            'epl-1.0', 'epl-2.0', 'etalab-2.0', 'eupl-1.1', 'agpl-3.0', 'gfdl', 'gpl', 'gpl-2.0',
            'gpl-3.0', 'lgpl', 'lgpl-2.1', 'lgpl-3.0', 'isc', 'lppl-1.3c', 'ms-pl', 'apple-ascl',
            'mpl-2.0', 'odc-by', 'odbl', 'openrail++', 'osl-3.0', 'postgresql', 'ofl-1.1', 'ncsa',
            'unlicense', 'zlib', 'pddl', 'lgpl-lr', 'deepfloyd-if-license', 'llama2', 'llama3',
            'llama3.1', 'llama3.2', 'gemma', 'unknown', 'other'
        ]

collections = {"es": "spanish-66f1460e7972f6224f479a17",
               "sv": "swedish-66f14687831eacfcad87bbb7",
               "ro": "romanian-66f14654c6592d7516edb1e9",
               "it": "italian-66f14649878c56e920b38fdb",
               "nl": "dutch-66f14641caf6968847a453a9",
               "cs": "czech-66f14639c46132b895c8ad55",
               "en": "english-66f14630ff0a35fed8e4c7de"}

repo_type = "model"

def description_text_model_norm(name, data_organisation, description, data_description,
                                language, license, tags, mod_type, mod_target):
    """
    Template for dataset card
    """
    minimal_tag_list = ['biomedical', 'lexical semantic', 'bionlp', 'biology', 'science', 'embedding', 'entity linking']
    tags = list(set(minimal_tag_list).union(set(tags)))

    metadata = f"""
---
id: {name}
name: {name}
description: {description}
license: {license}
language: {language}
tags: {tags}
pipeline_tag: feature-extraction
---
"""

    if mod_type == 'cls':
        mod_string = 'cls_rep = model(**toks_cuda)[0][:,0,:]'
    elif mod_type == 'mean':
        mod_string = 'cls_rep = model(**toks_cuda)[0].mean(1)'

    mod_requirements = f'''
### Expected input and output
The input should be a string of biomedical entity names, e.g., "covid infection" or "Hydroxychloroquine". The [CLS] embedding of the last layer is regarded as the output.

#### Extracting embeddings from {name}

The following script converts a list of strings (entity names) into embeddings.
```python
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("{data_organisation}/{name}")
model = AutoModel.from_pretrained("{data_organisation}/{name}").cuda()

# replace with your own list of entity names
all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"]

bs = 128 # batch size during inference
all_embs = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i+bs],
                                       padding="max_length",
                                       max_length=25,
                                       truncation=True,
                                       return_tensors="pt")
    toks_cuda = {"{}"}
    for k,v in toks.items():
        toks_cuda[k] = v.cuda()
    {mod_string}
    all_embs.append(cls_rep.cpu().detach().numpy())

all_embs = np.concatenate(all_embs, axis=0)
```
'''


    text = f"""{metadata}
# Model Card for {" ".join(name.split("_")).title()}

The model was trained on medical entity triplets (anchor, term, synonym)

{mod_requirements}

# Data description

{data_description}


# Acknowledgement

This is part of the [DT4H project](https://www.datatools4heart.eu/).

# Doi and reference


{mod_info_dict['sapbert'] if mod_target == 'sap' else mod_info_dict['mirrorbert']}

"""
    return text

######################################
######################################

def description_text_model_ner(name, data_organisation, description, data_description,
                                language, license, tags, mod_type, base_model, ner_classes):
    """
    Template for dataset card
    """
    minimal_tag_list = ['biomedical', 'lexical semantic', 'bionlp', 'biology', 'science', 'clinical ner', 'span classification']
    tags = list(set(minimal_tag_list).union(set(tags)))

    metadata = f"""
---
id: {name}
name: {name}
description: {description}
license: {license}
language: {language}
tags: {tags}
base_model : {base_model}
pipeline_tag: token-classification
---
"""
    if mod_type == 'multiclass':
        mod_specific = f'{name} is a multilabel-multiclass span classification model.'
    else:
        mod_specific = f'{name} is a muticlass span classification model.'

    #  This specific model is the average of the best checkpoints per fold over a ten-fold cross-validation over 547 labeled cardiology discharge letters
    mod_requirements = f'''

This a {base_model} base model finetuned for span classification. For this model
we used IOB-tagging. Using the IOB-tagging schema facilitates the aggregation of predictions
over sequences. This specific model is trained on a batch of 240 span-labeled documents.

### Expected input and output
The input should be a string with **Dutch** cardio clinical text.

{mod_specific}
The classes that can be predicted are {ner_classes}.

#### Extracting span classification from {name}

The following script converts a string of <512 tokens to a list of span predictions.
```python
from transformers import pipeline

le_pipe = pipeline('ner',
                    model=model,
                    tokenizer=model, aggregation_strategy="simple",
                    device=-1)

named_ents = le_pipe(SOME_TEXT)
```

To process a string of arbitrary length you can split the string into sentences or paragraphs
using e.g. pysbd or spacy(sentencizer) and iteratively parse the list of with the span-classification pipe.

'''


    text = f"""{metadata}
# Model Card for {" ".join(name.split("_")).title()}


{mod_requirements}

# Data description

{data_description}


# Acknowledgement

This is part of the [DT4H project](https://www.datatools4heart.eu/).

# Doi and reference


{mod_info_dict['cardioner']}

"""
    return text
