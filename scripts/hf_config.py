mod_info ='''
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

def description_text_model(name, description, data_description, language, license, tags):
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

tokenizer = AutoTokenizer.from_pretrained("UMCU/{name}")
model = AutoModel.from_pretrained("UMCU/{name}").cuda()

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
    cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
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


{mod_info}

"""
    return text