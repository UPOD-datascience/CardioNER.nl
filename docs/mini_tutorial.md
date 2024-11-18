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

The first thing we want to do is collect .ann's per document-id, so for ```casos_clinicos_cardiologia3``` we want to
have one .ann. To do this you e.g. do ```python Pubscience\pubscience\share\collect_ann.py --basefolder \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE --classname dis med proc symp```, 
this will create an ```ann``` folder with .ann's in ```\some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE```. We also want to have one folder with the txts, in principle the ```\txt``` folders 
for the classes should be identical so you can just copy the contents of anyone of them. Perhaps a good sanity check to see if the .txt folders are the same for each class. We want to end up with 
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
					dt4h_cardioccc_annotation_transfer_sv_symp.tsv
				\txt
					casos_clinicos_cardiologia3.txt
					casos_clinicos_cardiologia10.txt
					...	
```
Now, we can create a single .jsonl, just do ```python Pubscience\pubscience\share\ner_caster.py --txt_dir \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE\txt --ann_dir \some_root_folder\b1\2_validated_w_sugs\YOUR_LANGUAGE\ann --out_path OUTPUT_LOCATION```.