"""
Incoming format is :
[
{"tags": [{"start": xx, "end":xx, "tag": "DISEASE"},...],
 "id": xxx,
 "text": xxx}

]
"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from collections import defaultdict, Sequence

from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

class DataCollator():
    def __init__(self, labels: List[str]=['DIS', 'PROC', 'SYMP', 'MED']):
        self.labels = labels
        self.label2id = {l:c for c,l in enumerate(labels)}
        self.id2label = {c:l for l,c in self.label2id.items()}


    def align_tags_with_tokens(tags: List[Dict],
                           ):



