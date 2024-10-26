# CardioNER.nl

We develop a NER-model for the cardiology domain based on manually annotated translated documents as part of a multilingual NLP-effort.
The model is validated on original Dutch EHR documents from the Amsterdam UMC.

We use different models: 
* Transformer
 * RobBERTv2
 * MedRoBERTa.nl -> _baseline model for the Datatools4Heart project_, via [simpletransformers](https://simpletransformers.ai/docs/ner-model/)?
* CNN
 * 1D CNN
 * TextCNN
* GCN
 * TextGCN


# Data structure

```
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
```
T5	SYMPTOM 2508 2525	febriele syndroom
T1	SYMPTOM 2794 2850	HEMOCULTUREN: 23/07: positief voor Staphylococcus aureus
T7	SYMPTOM 3127 3182	cardiothoracale index (CTI) op de grens van normaliteit
```

The ```.txt``` files contain the original text, as-is.

The ```.tsv``` file is structured as follows.

```
name	tag	start_span	end_span	text	note
casos_clinicos_cardiologia286	SYMPTOM	109	116	dyspneu	
casos_clinicos_cardiologia286	SYMPTOM	120	151	oedeem in de onderste ledematen	
casos_clinicos_cardiologia286	SYMPTOM	1055	1099	progressieve dyspneu tot minimale inspanning	
casos_clinicos_cardiologia286	SYMPTOM	1103	1134	oedeem in de onderste ledematen	
casos_clinicos_cardiologia286	SYMPTOM	1214	1223	orthopnoe	
casos_clinicos_cardiologia286	SYMPTOM	1227	1258	paroxismale nachtelijke dyspnoe	
casos_clinicos_cardiologia286	SYMPTOM	1591	1617	Algemene conditie behouden
...
```

For the purpose of training transformer-based models using the ```transformers``` library we would like to recast the
datastructure into a JSONL in the following format:
```
[
    {
        'id': ,
        'tokens': [],
        'pos_tags': [],
        'chunk_tags': [],
        'ner_tags': [],
        'annotation_batch': 'b1',
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

This can be directly loaded into huggingface datasets.
