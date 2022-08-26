# coding: utf-8
import argparse
import time
import math
import os
import pickle

import numpy as np
import torch

from manager import get_data

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
unfinished_set = None

device = args.device

def get_sentences(id_tensor, min_length=4, vocab=None):
    """ Load sentences """

    global unfinished_set

    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor) or isinstance(id_tensor, torch.Tensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)

    # Continue from the last partial sentence.
    set_ = unfinished_set or []
    sets = []
    for i in ids:
        set_.append(int(i))
        if int(i) == 9:
            if len(set_) >= min_length:
                sets.append(set_)
            set_ = []

    # Handling a partial sentence on the last of a batch.
    if len(set_) != 0:
        unfinished_set = set_
    return sets

##############################################################################
# Load data
###############################################################################

train_iters, val_iters, test_iters = get_data(args.data, args.batch_size, args.bptt, args.device)
ntokens = len(train_iters.dataset.fields["text"].vocab)

# make_docs
docs = []
for batch, item in enumerate(train_iters):
    data = item.text
    batch_ = get_sentences(data, vocab=train_iters.dataset.fields["text"].vocab)
    docs.append(batch_) # batchify

import importlib.util
spec = importlib.util.spec_from_file_location("compute_tfidf", "../../tools/compute_tfidf.py")
compute_tfidf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_tfidf)
compute_tfidf.compute(ntokens, docs, args.data, args.mode, alpha=0.0, beta=0.1, base=8)
