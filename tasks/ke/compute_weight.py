# coding: utf-8
import argparse
import time
import math
import os
import pickle

import numpy as np
import torch

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

dataset = "FB15K"

if dataset == "FB15K":
    train_path = "third_party/benchmarks/FB15K/train2id.txt"
    entity_path = "third_party/benchmarks/FB15K/entity2id.txt"
    relation_path = "third_party/benchmarks/FB15K/relation2id.txt"

with open(entity_path, "r") as f:
    num_entity = int(f.readline())

with open(relation_path, "r") as f:
    num_relation = int(f.readline())

e2f = [ 0 for _ in range(num_entity) ]
r2f = [ 0 for _ in range(num_relation) ]
with open(train_path, "r") as f:
    cnt = int(f.readline()) # num of instances
    for _ in range(cnt):
        line = f.readline()
        tokens = tuple([int(x) for x in line.split(" ")])
        h, t, r = tokens

        e2f[h] += 1
        e2f[t] += 1
        r2f[r] += 1

score = (e2f, r2f)
with open("score.pkl", "wb") as f:
    pickle.dump(score, f)
