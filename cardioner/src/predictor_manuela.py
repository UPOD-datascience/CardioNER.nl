from __future__ import absolute_import, division, print_function
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (BertConfig, BertTokenizer)
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from pytorch_transformers import BertForTokenClassification
from torch import nn
import os
import re
import pandas as pd
from tqdm import tqdm
import spacy


# nlp = spacy.load('ro_core_news_lg') # Romanian
# nlp = spacy.load('xx_sent_ud_sm') # Multilingual
# nlp = spacy.load('sv_core_news_lg') # Swedish
# nlp = spacy.load('it_core_news_lg') # Italian
# nlp = spacy.load('es_core_news_lg') # Spanish
nlp = spacy.load('en_core_web_lg') # English
# nlp = spacy.load('nl_core_news_lg') # Dutch


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logging.disable(logging.INFO)
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")


class Ner(BertForTokenClassification):

    def forward(self, input_ids, device, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, valid_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_id = label_id
        self.valid_ids = valid_ids
        # self.label_mask = label_mask


def readfile(filename):
    f = open(filename, encoding="utf8")
    data = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append(sentence)
                sentence = []
            continue
        splits = line.split('\n')
        sentence.append(splits[0])

    if len(sentence) > 0:
        data.append(sentence)

    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, data_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, data_file)), "train")

    def get_dev_examples(self, data_dir, data_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, data_file)), "dev")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, sentence in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        textlist = example.text_a.split(' ')
        tokens = []
        valid = []
        for word in textlist:
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid.append(1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, valid_ids=valid))
    return features


def get_token_indices(input_string, delimiter=' '):
    # Split the input string into tokens
    tokens = input_string.split(delimiter)
    
    # Initialize variables to keep track of the start index
    current_index = 0
    token_indices = []
    
    for token in tokens:
        # Find the start index of the current token
        start_index = input_string.find(token, current_index)
        token_indices.append((token, start_index))
        
        # Update the current index to be just after this token's start index
        current_index = start_index + len(token)
    
    return token_indices


def count_leading_whitespaces(input_string):
    match = re.match(r'^\s*', input_string)
    if match:
        return len(match.group())
    return 0


def append_to_tsv(df, file_path):
    # Check if the file already exists
    if os.path.exists(file_path):
        # If the file exists, append without writing the header
        df.to_csv(file_path, mode='a', index=False, sep='\t', header=False)
    else:
        # If the file does not exist, write the DataFrame with the header
        df.to_csv(file_path, mode='w', index=False, sep='\t', header=True)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert_models/bert-base-multilingual-cased-ner-hrl', type=str,
                        help="Bert pre-trained model selected in the list:"
                             "bert-base-romanian-ner (31), "
                             "bert-base-NER (9), "
                             "bert-base-swedish-cased-ner (14), "
                             "bert-italian-finetuned-ner (33), "
                             "bert-spanish-cased-finetuned-ner (9), "
                             "xlm-ner-slavic (9), "
                             "bert-base-multilingual-cased-ner-hrl (9). ")
    parser.add_argument("--task_name", default='ner', type=str, help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval or not.")
    parser.add_argument("--eval_on", default="test", help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=9e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.15, type=float, help="attention_probs_dropout_prob")
    parser.add_argument("--hidden_dropout_prob", default=0.15, type=float, help="hidden_dropout_prob")
    parser.add_argument("--hidden_act", default='gelu', type=str, help="hidden_act")
    parser.add_argument("--num_attention_heads", default=16, type=int, help="num_attention_heads")
    parser.add_argument("--num_hidden_layers", default=16, type=float, help="num_hidden_layers")
    parser.add_argument("--num_train_epochs", default=10, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.000, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-7, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--no_cuda", action='store_false', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()


    model_dir = 'english_multilingual_disease/checkpoint_5'
    # label_list = ["O", "B-MEDICATION", "I-MEDICATION", "[CLS]", "[SEP]"]
    # label = "MEDICATION"
    label_list = ["O", "B-DISEASE", "I-DISEASE", "[CLS]", "[SEP]"]
    label = "DISEASE"
    path_list = ['../data/CardioCCC_Dataset/Testing/English/txt']
    output_file = 'english_multilingual_disease/inf_results_on_en_testset_disease.tsv'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = NerProcessor()
    num_labels = 9
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.hidden_act = args.hidden_act
    config.num_attention_heads = args.num_attention_heads
    config.num_hidden_layers = args.num_hidden_layers
    model = Ner.from_pretrained(model_dir, from_tf=False, config=config)
    model.to(device)
    model.eval()

    for path in path_list:
        for file_idx, file in tqdm(enumerate(os.listdir(path))):
            print(" ", file_idx, file)
            if file.endswith(".txt"):
                df = pd.DataFrame(columns = ['filename', 'label', 'start_span', 'end_span', 'text'])
                report_text_file = file
                with open(os.path.join(path, report_text_file), "r", encoding='utf8') as myfile:
                    report_text = myfile.readlines()
                    report_text = ''.join(report_text)
                full_stop_positions = [x.start() for x in re.finditer('\.', report_text)]
                full_stop_positions = [0] + full_stop_positions + [len(report_text)]

                if len(full_stop_positions) > 0:
                    for position_index in range(1, len(full_stop_positions)):
                        sentence_start = full_stop_positions[position_index - 1]
                        sentence_end = full_stop_positions[position_index]
                        text = report_text[sentence_start:sentence_end + 1]

                        doc = nlp(text)
                        token_list = []
                        idx_list = []
                        is_punct = []
                        for token in doc:
                            if token.text=='.':
                                if token.idx!=0:
                                    token_list.append(token.text.strip())
                                    idx_list.append(token.idx)
                                    is_punct.append(token.is_punct)
                            elif token.text.replace('\n', '') != '':
                                token_list.append(token.text.strip())
                                idx_list.append(token.idx)
                                is_punct.append(token.is_punct)

                            
                        out_file = 'test.tsv'
                        with open(out_file, 'w', encoding='utf8') as output:
                            for token in token_list:
                                output.write(token + '\n')

                        # os.remove(out_file)

                        ### inference
                        eval_examples = processor.get_dev_examples('.', out_file)
                        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)
                        logger.info("***** Running evaluation test*****")
                        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_valid_ids)
                        # Run prediction for full data
                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                        
                        y_pred = []
                        label_map = {i: label for i, label in enumerate(label_list, 1)}
                        for input_ids, input_mask, segment_ids, valid_ids in eval_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            valid_ids = valid_ids.to(device)
                            with torch.no_grad():
                                logits = model(input_ids, device, segment_ids, input_mask, valid_ids=valid_ids)
                            logits = torch.argmax(F.log_softmax(logits[:, :, 1:len(label_list)-1], dim=2), dim=2) + 1
                            logits = logits.detach().cpu().numpy()

                            for i, logit in enumerate(logits):
                                temp = []
                                for j, _ in enumerate(logit):
                                    if j==0:
                                        continue
                                    elif logits[i][j] == len(label_map):
                                        y_pred.append(temp)
                                        break
                                    else:
                                        temp.append(label_map[logits[i][j]])

                        
                        y_pred = temp[:len(token_list)]
                        ## end inference

                        # for idx, (token, token_idx, prediction) in enumerate(zip(token_list, idx_list, y_pred)):
                        #     print(str(idx).ljust(17), token.ljust(17), str(token_idx + sentence_start).ljust(17), str(token_idx + len(token) + sentence_start).ljust(17), prediction.ljust(17))
                       
                        new_token = 1
                        entity_index = -1
                        last_entity_index = -1
                        for idx, (token, token_idx, is_punctuation, prediction) in enumerate(zip(token_list, idx_list, is_punct, y_pred)):
                            if label in prediction:
                                if new_token and is_punctuation:
                                    continue
                                else:
                                    start_entity_span = token_idx + sentence_start
                                    end_entity_span = token_idx + len(token) + sentence_start
                                    entity = token
                                    entity_index = idx
                                    last_entity_index = idx
                                    new_token = 0
                                    break
                        
                        if entity_index >= 0:
                            for idx, (token, token_idx, is_punctuation, prediction) in enumerate(zip(token_list[entity_index + 1:], idx_list[entity_index + 1:], is_punct[entity_index + 1:], y_pred[entity_index + 1:])):
                                if label in prediction:
                                    if new_token:
                                        if is_punctuation:
                                            continue
                                        else:
                                            start_entity_span = token_idx + sentence_start
                                            end_entity_span = token_idx + len(token) + sentence_start
                                            entity = token
                                            new_token = 0
                                            last_entity_index = idx + entity_index + 1
                                    else:
                                        start_span = token_idx + sentence_start
                                        end_span = token_idx + len(token) + sentence_start
                                        if start_span == end_entity_span + 1:
                                            end_entity_span = end_span
                                            entity = entity + ' ' + token
                                            last_entity_index = idx + entity_index + 1
                                        elif start_span == end_entity_span:
                                            end_entity_span = end_span
                                            entity = entity + token
                                            last_entity_index = idx + entity_index + 1
                                        else:
                                            if len(entity) >= 2:
                                                if entity[-2] == ' ' and is_punct[last_entity_index] and '(' not in entity[:-2]:
                                                    entity = entity[:-2]
                                                    end_entity_span = end_entity_span - 2
                                            if len(entity) >=1 :
                                                if entity[-1] ==')' and '(' not in entity:
                                                    entity = entity[:-1]
                                                    end_entity_span = end_entity_span - 1
                                            if entity.startswith(" "):
                                                entity = entity[1:]
                                                start_entity_span = start_entity_span + 1
                                            if entity.startswith("de "):
                                                entity = entity[3:]
                                                start_entity_span = start_entity_span + 3

                                            if entity.strip() != '' and entity!='de' and not entity.isnumeric():
                                                pred_dict = {'filename': file.split('.')[0], 
                                                            'label': label,
                                                            'start_span': start_entity_span,
                                                            'end_span': end_entity_span,
                                                            'text': entity
                                                            }
                                                df = df._append(pred_dict, ignore_index=True)

                                            new_token = 1
                                            if new_token:
                                                if is_punctuation:
                                                    continue
                                                else:
                                                    start_entity_span = start_span
                                                    end_entity_span = end_span
                                                    entity = token
                                                    new_token = 0
                                                    last_entity_index = idx + entity_index + 1

                            if len(entity) >= 2:
                                if entity[-2] == ' ' and is_punct[last_entity_index] and '(' not in entity[:-2]:
                                    entity = entity[:-2]
                                    end_entity_span = end_entity_span - 2
                            if len(entity) >=1 :
                                if entity[-1] ==')' and '(' not in entity:
                                    entity = entity[:-1]
                                    end_entity_span = end_entity_span - 1
                            if entity.startswith(" "):
                                entity = entity[1:]
                                start_entity_span = start_entity_span + 1
                            if entity.startswith("de "):
                                entity = entity[3:]
                                start_entity_span = start_entity_span + 3
                            if entity.strip() != '' and entity!='de' and not entity.isnumeric():
                                pred_dict = {'filename': file.split('.')[0], 
                                                        'label': label,
                                                        'start_span': start_entity_span,
                                                        'end_span': end_entity_span,
                                                        'text': entity
                                                        }
                                df = df._append(pred_dict, ignore_index=True)

                        os.remove(out_file)

                append_to_tsv(df, output_file)


if __name__=='__main__':
    main()