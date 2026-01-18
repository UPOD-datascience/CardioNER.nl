# CardioNER.nl

We develop a NER-model for the cardiology domain based on manually annotated translated documents as part of a multilingual NLP-effort.
The model is validated on original Dutch EHR documents from the Amsterdam UMC.

We use different models, for each language a BERT/RoBERTa/DeBERTa type model that


For the base-models we have 2 basic refinements beyond the vanilla finetuning:
* Adding a CRF to the head: a conditional random field can help in 'respecting' the IOB tag order.
* self-aligned pre-training (sap) on UMLS term/synonym pairs. Alternatives are BIOSYN, KRISSBERT, BioLord, MirrorBERT.

CardioNER.[lang] will then have 3 version:
* **v1**: finetuning of pre-trained biomedical model
* **v2**: finetuning of pre-trained biomedical model, that is further pre-trained using self-aligned pre-training on UMLS
* **v3**: we add a CRF head.
* **v4**: we add multihead CRF compatible with the transformer library
* **v5**: we add custom heads 


# Data structure

```yaml
/assets/b1
        /nl
            dis
            med
            proc
            symp/
                ann/
                    casos_clinicos_cardiologia3.ann
                    casos_clinicos_cardiologia10.ann
                    ...
                tsv/
                    dt4h_cardioccc_annotation_transfer_nl_symp.tsv
                text/
                    casos_clinicos_cardiologia3.txt
                    casos_clinicos_cardiologia10.txt
                    ...
```


The file ```casos_clinicos_cardiologia3.ann``` has the following format
```tsv
T5	SYMPTOM 2508 2525	febriele syndroom
T1	SYMPTOM 2794 2850	HEMOCULTUREN: 23/07: positief voor Staphylococcus aureus
T7	SYMPTOM 3127 3182	cardiothoracale index (CTI) op de grens van normaliteit
```

The ```.txt``` files contain the original text, as-is.

The ```.tsv``` file is structured as follows.

```tsv
name	tag	start_span	end_span	text	note
casos_clinicos_cardiologia286	SYMPTOM	109	116	dyspneu
casos_clinicos_cardiologia286	SYMPTOM	120	151	oedeem in de onderste ledematen
casos_clinicos_cardiologia286	SYMPTOM	1055	1099	progressieve dyspneu tot minimale inspanning
casos_clinicos_cardiologia286	SYMPTOM	1103	1134	oedeem in de onderste ledematen
casos_clinicos_cardiologia286	SYMPTOM	1214	1223	orthopnoe
casos_clinicos_cardiologia286	SYMPTOM	1227	1258	paroxismale nachtelijke dyspnoe
casos_clinicos_cardiologia286	SYMPTOM	1591	1617	Algemene conditie behouden
```

For the purpose of training transformer-based models using the ```transformers``` library we would like to recast the
datastructure into a JSONL in the following format:
```json
[
    {
        "id": ,
        "tokens": [],
        "pos_tags": [],
        "chunk_tags" [],
        "ner_tags" [],
        "annotation_batch": "b1",
    },
    {
        ...
    },
    ...
]
```

with
```
id2labels = {0: 'disease', 1: 'medication', 2: 'procedure', 3: 'symptom'}
```

```bash
python ner_caster.py --ann_dir=b1/1_validated_without_sugs/it/dis/ann  --txt_dir=b1/1_validated_without_sugs/it/dis/txt --out_path=/path/to/assets
```
or
```bash
python ner_caster.py --db_path=b1/1_validated_without_sugs/it/dis/tsv/bla.tsv  --txt_dir=b1/1_validated_without_sugs/it/dis/txt --out_path=/path/to/assets
```


This can be directly loaded into huggingface datasets.

# Instructions

Example:

Suppose in ```annotations.jsonl```we have the annotations in the following format

```json
{
 "id": "casos_clinicos_cardiologia83",
 "tags": [{"start": 117, "end": 138, "tag": "DISEASE"}, {"start": 140, "end": 160, "tag": "DISEASE"},...
 "text": "Patiënte is bekend met een aortaklepstenose en een mitralisklepinsufficiëntie."
 },
 ...
```

and in ```splits.json```we have
```json
{
"en":{
   "train":{
      "symp":[
         "casos_clinicos_cardiologia3",
         "casos_clinicos_cardiologia10",
         ...
      ]
   }
}
}
```

Then, to parse the annotations and train the multilabel model, we can run the following command for the Dutch language:
```bash
poetry shell
cd cardioner/src
python main.py --lang nl --Corpus_train /location/of/annotations.jsonl --split_file /location/of/splits.json --parse_annotations --train_model --max_token_length 64 --batch_size 32 --chunk_size 64 --chunk_type centered
```

To train a multiclass model, simply add ```--multi_class``` to the command.

To run with CPU (handier for debugging for e.g. tensor mismatches), prepend ```CUDA_VISIBLE_DEVICES=``` to the command.
So,
```bash
CUDA_VISIBLE_DEVICES="" python main.py --lang nl --Corpus_train /location/of/annotations.jsonl --split_file /location/of/splits.json --parse_annotations --train_model --max_token_length 64 --batch_size 32 --chunk_size 64 --chunk_type centered
```

The languages are referred to as:
```
'es' : Spanish,
'nl' : Dutch,
'en' : English,
'it' : Italian,
'ro' : Romanian,
'sv' : Swedish,
'cz' : Czech
```

# Lightning code by Lorenzo

This code will run faster, and uses paragraph splitting, but does not use IOB-tagging (yet).

You can also run the ```light_ner.py``` script.
```python
python light_ner.py --batch_size=8 --patience=5 --num_workers=4 --max_epochs=1 --root_path=/path/to/data --lang=it --devices=0 --model=IVN-RIN/bioBIT --output_dir /output/path
```

This will train a model and store a HuggingFace version in ```--output_dir```.

To test a model, you can run the following command:

```
poetry shell
cd cardioner/src
python test.py --model /location/of/model --lang nl --ignore_zero
```

To run the validation on a folder of jsonl's, e.g.
```
python run_validation.py --model=/media/bramiozo/Storage2/DATA/NER/DT4H_results/CardioBerta_clinical/lightning_medical_20epochs --lang=nl --ignore_zero --input_dir=../../assets/b2/1_validated_without_sugs/nl
```


To push a model to Huggingface, e.g.
```
python3 push_to_huggingface.py --data_organization=DT4H-IE --repo_id=DT4H-IE/CardioBERTa.nl_clinical_NL_MED --path_to_file=/path/to/model --name="CardioNER model --medication" --description="Finetuned CardioBERTa.nl model for detection of medication spans. This model is a mulilabel model using BCE loss." --data_description="50/50 Train/validation split on CardioCCC, a manually labeled cardiology corpus" --mod_type=multilabel --mod_target=ner --base_model="UMCU/CardioBERTa.nl_clinical" --ner_classes medication --branch=2025-05-14_0001 --license=mit --language=nl
```


Training:

**Multilabel**
```
python main.py \
    --lang nl \
    --corpus_train /path/to/train \
    --entity_types DRUG DISEASE SYMPTOM \
    --train_model \
    --parse_annotations \
    --output_dir ./output_multihead
```

**Multiclass** (*assumes no over span overlap!*)
```
python main.py \
    --lang nl \
    --corpus_train /path/to/train \
    --multiclass \
    --entity_types DRUG DISEASE SYMPTOM \
    --train_model \
    --parse_annotations \
    --output_dir ./output_multihead
```

**Multiclass CRF** (*assumes no over span overlap!*)
```
python main.py \
    --lang nl \
    --corpus_train /path/to/train \
    --multiclass \
    --use_crf \
    --entity_types DRUG DISEASE SYMPTOM \
    --train_model \
    --parse_annotations \
    --output_dir ./output_multihead
```

**Multiclass multihead**
```
python main.py \
    --lang nl \
    --corpus_train /path/to/train \
    --use_multihead \
    --entity_types DRUG DISEASE SYMPTOM \
    --train_model \
    --parse_annotations \
    --output_dir ./output_multihead
```

**Multiclass multihead CRF**
```
python main.py \
    --lang nl \
    --corpus_train /path/to/train \
    --use_multihead_crf \
    --entity_types DRUG DISEASE SYMPTOM \
    --train_model \
    --parse_annotations \
    --output_dir ./output_multihead
```
