# coding: utf-8
import argparse
import time
import math
import os
import pickle

import numpy as np
import torch

from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='wikitext2',
                    help='data corpus')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--mode', type=str, default="tfidf", help="mode")
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device')

args = parser.parse_args()

device = args.device

train_dataloader = TrainDataLoader(
	in_path = "./third_party/benchmarks/FB15K237/",
	nbatches = 100,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0)

num_entity = train_dataloader.get_ent_tot()

e2f = [ 0 for _ in range(num_entity) ]

for data in train_dataloader:
    for d in data['batch_h']:
        e2f[d] += 1
    for d in data['batch_t']:
        e2f[d] += 1

m_ = max(e2f)

score = []
for v in e2f:
    score.append(float(v+1) / m_)
with open("score.pkl", "wb") as f:
    pickle.dump(score, f)
