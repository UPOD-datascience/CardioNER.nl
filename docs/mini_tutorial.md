# Mini-tutorial

We start with the following file/folder structure

```
\some_root_folder
	\b1
		\0_brat_originals
		\1_validated_without_sugs
		\2_validated_w_sugs
			\cz
			\en
			\es
			\it
			\nl
			\ro
			\sv
				\dis
				\med
				\proc
				\symp
					\ann
						casos_clinicos_cardiologia3.ann
						casos_clinicos_cardiologia10.ann
						...
					\tsv
						dt4h_cardioccc_annotation_transfer_sv_symp.tsv
					\txt
						casos_clinicos_cardiologia3.txt
						casos_clinicos_cardiologia10.txt
						...
```
Note: we only care about ```2_validated_w_sugs```.

# Step 1
>[!TIP]
> before you continue
>
> do ```poetry install```
>
> do ```poetry shell```

The first thing we want to do is collect .ann's per document-id, so for ```casos_clinicos_cardiologia3``` we want to
have one .ann. To do this you e.g. do ```python Pubscience\pubscience\share\collect_ann.py --basefolder \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE --classname dis med proc symp```,
this will create an ```ann``` folder with .ann's in ```\some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE```. We also want to have one folder with the txts, in principle the ```\txt``` folders
for the classes should be identical so you can just copy the contents of anyone of them. Perhaps a good sanity check to see if the .txt folders are the same for each class. To create one .tsv, just concat the .tsv's, but once you have created the single ann-folder this is not really necessary.
We want to end up with
```
\some_root_folder
	\b1
		\2_validated_w_sugs
			\cz
			\en
			\es
			\it
			\nl
			\ro
			\sv
				\ann
					casos_clinicos_cardiologia3.ann
					casos_clinicos_cardiologia10.ann
					...
				\tsv
					dt4h_cardioccc_annotation_transfer_sv.tsv
				\txt
					casos_clinicos_cardiologia3.txt
					casos_clinicos_cardiologia10.txt
					...
```

## Output

Now, we can create a single .jsonl, just do ```python Pubscience\pubscience\share\ner_caster.py --txt_dir \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE\txt --ann_dir \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE\ann --out_path OUTPUT_LOCATION```.

OK, so, the .jsonl contains dictionaries as
```
  {'tags': [{'start':  xx, 'end': xx, 'tag': 'DISEASE'},
            {'start':  xx, 'end': xx, 'tag': 'PROCEDURE'}
            ...
  ],
  "id": "casos_clinicos_cardiologia83",
  "text": "bla.."
   }
```

# Step 2
But, the transformer model expects the following format
```
{
"id": "casos_clinicos_cardiologia83",
"input_ids": [132,4125,5234,2356,23,88,36,...],
"labels": [0,0,0,0,0,0,0,1,2,2,0,0,0,0,3,4,4,...]
}
```

Where the labels are represented as integers with some mapping, for instance
```
{0: 'O', 1: 'B-DISEASE', 2: 'I-DISEASE', 3: 'B-PROCEDURE', 4: 'I-PROCEDURE',..}
```

To achieve this we first tokenize with a standard splitter such as available in SpaCy, so we get lists of words, and for each word we have a tag, one of 'O', 'B-DISEASE'...'I-MEDICATION'. Long story short, we get a list with
```
{
 'id': ...,
 'tokens': [..],
 'tags': [..]
}
```

We are not done yet, unfortunately, these tokens and tags and not yet in the right form for our model. We need to replace the tokens obtained from our SpaCy splitter with the tokens resulting from the tokenizer that was used to created the pre-trained transformer model. These tokens are really the id's that refer to human-readable tokens in the vocabulary that make the tokenizer. This splitting of the earlier tokens from SpaCy into possibly multiple different tokens requires a re-alignment of the tags. With some python magic and use of the pre-trained tokenizer we end up with our desired form;
```
{
"id": "casos_clinicos_cardiologia83",
"input_ids": [132,4125,5234,2356,23,88,36,...],
"labels": [0,0,0,0,0,0,0,1,2,2,0,0,0,0,3,4,4,...]
}
```

## Output
...OK we are not really done :D. The length of the input_ids can exceed the maximum input length for our models. To account for this we need to split our inputs. I have now done this in the SpaCy text-splitting step, which is suboptimal..but for now it works. The benefit of this approach is that it much easier to e.g. respect sentence boundaries, although strictly speaking we can also identify sentence delimiters from the tokenization vocabulary. Anyways, this is a potential improvement. Continuing, we end up with entries like

```
{
"gid": "casos_clinicos_cardiologia83",
"batch": "b1",
"id": "casos_clinicos_cardiologia83_spanX"
"input_ids": [132,4125,5234,2356,23,88,36,...],
"labels": [0,0,0,0,0,0,0,1,2,2,0,0,0,0,3,4,4,...]
}
```

To create the jsonl with this format, run
```
python main.py --parse_annotations --Corpus_b1 /loc/of/step1_b1.jsonl --Corpus_b2 /loc/of/step1_b2.jsonl --annotation_loc /loc/of/final.jsonl --chunk_size xx --chunk_type centered --Model YOUR_MODEL_LOCATION
```

Where ```chunk_size``` is the number of words (not tokens) per chunk, and ```chunk_type``` can be centered or standard. Centered chunking does just what it says, the labeled span is centered in the context window. Each span is represented by its own document as it where. Default chunking on the other hand just splits up the original document in chunks of xx words. The benefit of the latter is that we end up with a small multiple of the original number of documents versus the almost 100-fold increase if we create a new document around each span. The benefit of the former is that the model is more agnostic with regard to the location of the span in the document. We can try both.

# Step 3, training

To train simply do
```
python main.py --train_model --annotation_loc /loc/of/final.jsonl --num_splits 10
```

A possible improvement here: add location of ```model_settings.yaml``` to parse the arguments for the model training, now it is hard coded.. Also we need arguments for the locations of the splits.
