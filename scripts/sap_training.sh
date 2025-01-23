set -o allexport
source .env
set +o allexport

CUDA_VISIBLE_DEVICES=0 python sap_training.py \
	--model_dir "CLTL/MedRoBERTa.nl" \
	--train_dir $PATH_TO_TRAIN_FILE \
	--output_dir tmp \
	--use_cuda \
	--epoch 1 \
	--train_batch_size 256 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 999999 \
	--parallel \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"
