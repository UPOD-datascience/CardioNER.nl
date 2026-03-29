"""
Script to deploy Lang Extract:
    - scaffolding to implement examples
    - scaffolding to implement prompts
"""

import argparse
from typing import List

import langextract as lx
from pydantic import BaseModel

from cardioner.llm import lx_extract, lx_prompts


# https://github.com/google/langextract
# https://abdullah-humayun.medium.com/part-1-langextract-googles-new-named-entity-recognition-ner-model-0dffcb2692e4
def parse_examples(
    example_json: str, examples: List[dict] | None = None
) -> lx.data.ExampleData:
    pass


# use merging strategy, then extract, then unwind
# MERGE: concatenate multiple texts
# EXTRACT: get entities with their spans, use multiple passes and multiple workers
# UNWIND: map global spans back to correct documentwise spans
def extract(
    txts: List[str],
    Examples: lx.data.ExampleData,
    batch_size: int = 128,
    batch_len_max: int = 64_000,
) -> List[dict]:
    pass
