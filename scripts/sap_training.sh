set -o allexport
source .env
set +o allexport

CUDA_VISIBLE_DEVICES=0 python sap_training.py \
	--model_dir "CLTL/MedRoBERTa.nl" \
	--train_dir $PATH_TO_TRAIN_FILE \
	--output_dir tmp \
	--epoch 1 \
	--train_batch_size 64 \
	--learning_rate 1e-5 \
	--max_length 50 \
	--checkpoint_step 1000 \
	--pairwise \
	--num_workers 1 \
	--random_seed 7 \
	--use_miner \
	--loss ms_loss \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"
