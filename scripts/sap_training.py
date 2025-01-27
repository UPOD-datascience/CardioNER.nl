# CODE FROM https://github.com/cambridgeltl/sapbert/tree/main


#!/usr/bin/env python
import numpy as np
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import os
import json
import random
from tqdm import tqdm
import sys
sys.path.append("sap_src")
import wandb
from sap_src.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_custom,
    QueryDataset_pretraining,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
)
from sap_src.model_wrapper import Model_Wrapper
from sap_src.metric_learning import Sap_Metric_Learning

LOGGER = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description='sapbert train')
    parser.add_argument('--model_dir', help='Directory for pretrained model')
    parser.add_argument('--train_dir', type=str, required=True, help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output')
    parser.add_argument('--max_length', default=25, type=int)
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--train_batch_size', default=240, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--amp', action="store_true", help="automatic mixed precision training")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--pairwise', action="store_true", help="if loading pairwise formatted datasets")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--random_seed', default=1996, type=int)
    parser.add_argument('--loss', default="ms_loss")
    parser.add_argument('--use_miner', action="store_true")
    parser.add_argument('--miner_margin', default=0.2, type=float)
    parser.add_argument('--type_of_triplets', default="all", type=str)
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_all_tok}")
    parser.add_argument('--trust_remote_code', action="store_true", help="allow for custom models defined in their own modeling files")
    return parser.parse_args()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_dictionary(dictionary_path):
    dictionary = DictionaryDataset(dictionary_path=dictionary_path)
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(data_dir=data_dir, filter_composite=filter_composite, filter_duplicate=filter_duplicate)
    return dataset.data

def load_queries_pretraining(data_dir, filter_duplicate):
    dataset = QueryDataset_pretraining(data_dir=data_dir, filter_duplicate=filter_duplicate)
    return dataset.data

def collate_fn_batch_encoding(batch, tokenizer, max_length):
    query1, query2, query_id = zip(*batch)
    query_encodings1 = tokenizer.batch_encode_plus(
        list(query1),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    query_encodings2 = tokenizer.batch_encode_plus(
        list(query2),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    query_ids = torch.tensor(list(query_id))
    return query_encodings1, query_encodings2, query_ids

class CollateFnWrapper:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        return collate_fn_batch_encoding(batch, self.tokenizer, self.max_length)

def train(args, data_loader, model, tokenizer, scaler=None, model_wrapper=None, step_global=0):

    wandb.init(project="sapbert")

    LOGGER.info("train!")
    train_loss = 0
    train_steps = 0
    if args.use_cuda:
        model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {},{}
        for k,v in batch_x1.items():
            if args.use_cuda:
                batch_x_cuda1[k] = v.cuda()
            else:
                batch_x_cuda1[k] = v
        for k,v in batch_x2.items():
            if args.use_cuda:
                batch_x_cuda2[k] = v.cuda()
            else:
                batch_x_cuda2[k] = v
        if args.use_cuda:
            batch_y_cuda = batch_y.cuda()
        else:
            batch_y_cuda = batch_y
        if args.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()
        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model.encoder.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global

def model_wr(args):
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        trust_remote_code=args.trust_remote_code,
    )
    return tokenizer, encoder

def main(args, tokenizer=None, encoder=None):
    init_logging()
    torch.manual_seed(args.random_seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = Sap_Metric_Learning(
        encoder=encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        pairwise=args.pairwise,
        loss=args.loss,
        use_miner=args.use_miner,
        miner_margin=args.miner_margin,
        type_of_triplets=args.type_of_triplets,
        agg_mode=args.agg_mode,
    )
    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")

    collate_fn = CollateFnWrapper(tokenizer, args.max_length)
    if args.pairwise:
        train_set = MetricLearningDataset_pairwise(path=args.train_dir, tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        train_set = MetricLearningDataset(path=args.train_dir, tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.train_batch_size,
            sampler=samplers.MPerClassSampler(train_set.query_ids, 2, length_before_new_iter=100000),
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    scaler = GradScaler() if args.amp else None
    start = time.time()
    step_global = 0
    model_wrapper = Model_Wrapper()
    for epoch in range(1, args.epoch + 1):
        LOGGER.info("Epoch {}/{}".format(epoch, args.epoch))
        train_loss, step_global = train(args, data_loader=train_loader, model=model, tokenizer=tokenizer, scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss, epoch))
    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time! {} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))

if __name__ == '__main__':
    args = parse_args()

    if args.num_workers == 1:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    tokenizer, encoder = model_wr(args)
    main(args, tokenizer=tokenizer, encoder=encoder)
