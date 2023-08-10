#!/usr/bin/env python3
# Copyright (c) 
#        Meta Platforms, Inc. and affiliates.
#        Qiongkai Xu
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The following code is modified from sent140_tutorial for FLSim

import itertools
import json
import re
import string
import unicodedata
from typing import List
import os, sys
import pdb

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config

from utils.dataset_utils import (
    DataProvider,
    FLModel,
    LEAFDataLoader,
    LMMetricsReporter,
    newsgroup_labels,
    speaker_idxs_288 as speaker_idxs,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling, DefaultDataCollator
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AdamW
from torch.optim import SGD

from datasets import load_dataset


class DialogueDataset(Dataset):
    MIN_SAMPLE_NUM = 288 # at most 70 clients
    
    def __init__(self, tokenizer, max_seq_len, client_num=2, s_idx=0,e_idx=640):
        self.tokenizer = tokenizer
        self.max_seq_len = min(max_seq_len, tokenizer.model_max_length)
        self.client_num = client_num
        
        self.data = {}
        raw_dataset = load_dataset("empathetic_dialogues", split='train')
        
        # fs = []
        # for s_idx in speaker_idxs:
        #     tmp_data = raw_dataset.filter(lambda example, idx: example['speaker_idx']==s_idx, with_indices=True)
            
        #     if len(tmp_data) >= self.MIN_SAMPLE_NUM:
        #         fs.append(s_idx)
        # print(self.MIN_SAMPLE_NUM, len(fs))
        # print(fs)
        # exit()
            
        for speaker_idx in speaker_idxs[:client_num]:
            tmp_data = raw_dataset.filter(lambda example, idx: example['speaker_idx']==speaker_idx, with_indices=True)
            assert len(tmp_data) >= self.MIN_SAMPLE_NUM, 'Speaker {} without enough data'.format(speaker_idx)

            process_dataset = self.tokenize_function(tmp_data)
            process_dataset['input_ids'] = self.process_batch(process_dataset['input_ids'][s_idx:e_idx])
            process_dataset['attention_mask'] = self.process_batch(process_dataset['attention_mask'][s_idx:e_idx])
            self.data[speaker_idx] = process_dataset
        
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['utterance'], 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_len
            )
        
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data:
            raise IndexError(f"User {user_id} is not in dataset")
        return self.data[user_id]['input_ids'], self.data[user_id]['attention_mask']


    def extend_line(self, line_list: list[int], max_seq_len: int):
        indices = line_list + ([self.tokenizer.eos_token_id] * (max_seq_len - len(line_list)))
        return indices

    def process_batch(self, raw_batch):
        max_len = min(max([len(line) for line in raw_batch]), self.max_seq_len)
        batch = [self.extend_line(e, max_len) for e in raw_batch]
        batch = torch.LongTensor(batch)
        return batch

    def flatten_list(self, nested_list):
        return list(itertools.chain.from_iterable(nested_list))
    

def build_data_provider_dial(data_config, tokenizer, drop_last: bool = False):

    train_dataset = DialogueDataset(
        tokenizer = tokenizer,
        client_num = data_config.max_client_num,
        max_seq_len=data_config.max_seq_len,
        s_idx=0, e_idx=256
    )
    test_dataset = DialogueDataset(
        tokenizer = tokenizer,
        client_num = data_config.max_client_num,
        max_seq_len=data_config.max_seq_len,
        s_idx=256, e_idx=256+32
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=drop_last,
    )

    data_provider = DataProvider(dataloader)
    return data_provider


class NewsGroupDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len, client_num=2, s_idx=0,e_idx=640):
        self.tokenizer = tokenizer
        self.max_seq_len = min(max_seq_len, tokenizer.model_max_length)
        self.client_num = client_num
        newsgroup_topics = newsgroup_labels[0:client_num]
        
        self.data = {}
        for topic in newsgroup_topics:
            raw_dataset = load_dataset("newsgroup", topic)
            process_dataset = self.tokenize_function(raw_dataset['train'][s_idx:e_idx])
            
            process_dataset['input_ids'] = self.process_batch(process_dataset['input_ids'])
            process_dataset['attention_mask'] = self.process_batch(process_dataset['attention_mask'])
            self.data[topic] = process_dataset
            
            print(topic, len(raw_dataset['train']), process_dataset['input_ids'].size())
            
        
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'], 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_len
            )
        
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str):
        if user_id not in self.data:
            raise IndexError(f"User {user_id} is not in dataset")
        return self.data[user_id]['input_ids'], self.data[user_id]['attention_mask']

    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    def extend_line(self, line_list: list[int], max_seq_len: int):
        indices = line_list + ([self.tokenizer.eos_token_id] * (max_seq_len - len(line_list)))
        return indices

    def process_batch(self, raw_batch):
        max_len = min(max([len(line) for line in raw_batch]), self.max_seq_len)
        batch = [self.extend_line(e, max_len) for e in raw_batch]
        batch = torch.LongTensor(batch)
        return batch

    def flatten_list(self, nested_list):
        return list(itertools.chain.from_iterable(nested_list))
    
def build_data_provider_news(data_config, tokenizer, drop_last: bool = False):
    
    train_dataset = NewsGroupDataset(
        tokenizer = tokenizer,
        client_num = data_config.max_client_num,
        max_seq_len=data_config.max_seq_len,
        s_idx=0, e_idx=512
    )
    test_dataset = NewsGroupDataset(
        tokenizer = tokenizer,
        client_num = data_config.max_client_num,
        max_seq_len=data_config.max_seq_len,
        s_idx=512, e_idx=512+64
    )
    
    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=drop_last,
    )
    data_provider = DataProvider(dataloader)
    return data_provider


def main_worker(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available: bool = True,
    distributed_world_size: int = 1,
) -> None:
    
    if model_config.name.startswith('local-gpt2'):
        plm_dir = os.path.join(model_config.dir, model_config.name)
        print('reading from local model: ' + plm_dir)
        model = AutoModelForCausalLM.from_pretrained(plm_dir)
        tokenizer = AutoTokenizer.from_pretrained(plm_dir)
    elif model_config.name.startswith('scratch-gpt2'):
        plm_dir = os.path.join(model_config.dir, model_config.name)
        print('reading from scratch model: ' + plm_dir)
        from transformers import GPT2Config
        config = GPT2Config.from_pretrained(os.path.join(plm_dir, 'config.json'))
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        raise Exception('Unknown language model {}'.format(model_config.name))
    
    sys.path.append('/home/tools/Opacus-lab') # directory of opacus-lab
    from opacus_lab.models.GPT2.refactor import refactor_transformer
    size = 'T'
    print("Transfer GPT to fit Opacus")
    print(model)
    model = refactor_transformer(model, size=size)
    print(model)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if data_config.name == 'news':
        data_provider = build_data_provider_news(data_config, tokenizer)
    elif data_config.name == 'dial':
        data_provider = build_data_provider_dial(data_config, tokenizer)
    else:
        raise Exception('Dataset {} is not supported.'.format(data_config.name))
    
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    metrics_reporter = LMMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=distributed_world_size,
    )



@hydra.main(config_path=None, config_name="lm_config")
def run(cfg: DictConfig) -> None:
    
    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if trainer_config.client.store_models_dir is not None:
        exp_name = '{}_c{}_b{}_e{:.0f}_lr{}_{}'.format(model_config.name, 
                                                    data_config.max_client_num, 
                                                    data_config.local_batch_size, 
                                                    trainer_config.epochs, 
                                                    trainer_config.server.server_optimizer.lr, 
                                                    'sing' if trainer_config.client.one_batch_each_epoch else 'all')
        # for DP exps
        if 'privacy_setting' in trainer_config.client: 
            exp_name += '_c{}_n{:.0f}'.format(trainer_config.client.privacy_setting.noise_multiplier, trainer_config.client.privacy_setting.clipping.clipping_value)
        trainer_config.client.store_models_dir  = os.path.join(trainer_config.client.store_models_dir, data_config.name, exp_name)
        if not os.path.exists(trainer_config.client.store_models_dir):
            os.mkdir(trainer_config.client.store_models_dir )
        print("Save exp to: ", trainer_config.client.store_models_dir)
        sys.stdout = open(os.path.join(trainer_config.client.store_models_dir, 'output.txt'), 'w')
    print(OmegaConf.to_yaml(cfg))
    
    main_worker(trainer_config, model_config, data_config)


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
