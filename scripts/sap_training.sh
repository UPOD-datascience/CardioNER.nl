set -o allexport
source .env
set +o allexport

CUDA_VISIBLE_DEVICES=0 python sap_training.py \
	--model_dir "CLTL/MedRoBERTa.nl" \
	--train_dir $PATH_TO_TRAIN_FILE \
	--output_dir tmp \
	--epoch 2 \
	--use_cuda \
	--train_batch_size 64 \
	--learning_rate 5e-6 \
	--weight_decay 1e-4 \
	--max_length 30 \
	--checkpoint_step 10000 \
	--pairwise \
	--num_workers 2 \
	--random_seed 7 \
	--use_miner \
	--loss ms_loss \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "mean"
