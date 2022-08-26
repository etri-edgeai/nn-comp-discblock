# coding: utf-8
""" Running language modeling

"""
import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)

import argparse
import time
import math
import os

import torchtext
import torch.nn as nn
import yaml
import pickle

import copy

from tasks.lm.models.loader import load_model
from tasks.lm import manager

from discblock import magic
from config_parser import get_embedding_options

def evaluate_gates(magical_convert, config, model, dict_, manager, ntokens, val_iters, device):
    """ Evaluate gates """

    def apply_(model, dict_):
        model.load_state_dict(dict_)
        return model

    load_func = lambda : apply_(load_model(
        model_type=config["model"],
        ntokens=ntokens,
        ninp=config["emsize"],
        nhid=config["nhid"],
        nlayers=config["nlayers"],
        dropout=config["dropout"],
        nhead=config["nhead"],
        device=device), dict_)

    eval_func = lambda model: manager.evaluate(
        config["model"], model, data_source=val_iters, ntokens=ntokens, eval_batch_size=config["batch_size"])

    val_loss = magical_convert.evaluate_gates(model, config["block_options"], load_func, eval_func)
    print("SVD Test: ppl {:8.2f}".format(math.exp(val_loss)))
    return val_loss

parser = argparse.ArgumentParser(description='', add_help=False)
parser.add_argument('--config', type=str, default=None, help='configuration')
args = parser.parse_args()

with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Set the random seed manually for reproducibility.
torch.manual_seed(config["seed"])
device = torch.device(config["device"])

train_iters, val_iters, test_iters = manager.get_data(config["data"], config["batch_size"], config["bptt"], device)
ntokens = len(train_iters.dataset.fields["text"].vocab)

options = get_embedding_options(config)

magical_convert = magic.EmbeddingMagic(
    mode=config["mode"],
    embeddings=config["embeddings"],
    target_ratio=config["target_ratio"],
    embedding_type=config["embedding"],
    options=options,
    use_embedding_for_decoder=config["use_embedding_for_decoder"],
    device=device)

# load model
if config["mode"] == "test":
    model = torch.load(config["weights"], map_location="cpu")
    model.to(device)
else:
    if config["weights"] != "none":
        model = torch.load(config["weights"], map_location="cpu")
        model.to(device)
    else:
        model = load_model(
            model_type=config["model"],
            ntokens=ntokens,
            ninp=config["emsize"],
            nhid=config["nhid"],
            nlayers=config["nlayers"],
            dropout=config["dropout"],
            nhead=config["nhead"],
            device=device).to(device)

    if config["embedding"] != "none":
        if config["weights"] == "none":
            magical_convert.convert(model)
        else:
            dict_ = torch.load(config["weights"], map_location="cpu")
            if not isinstance(dict_, dict):
                dict_ = dict_.state_dict()
            model.load_state_dict(dict_)
            magical_convert.convert(model, setup_weights=True)

            model.to(device)
            val_loss = manager.evaluate(
                model_type=config["model"],
                model=model,
                data_source=val_iters,
                ntokens=ntokens,
                eval_batch_size=config["batch_size"])
            print("Before training {:8.2f} ppl".format(math.exp(val_loss)))

            if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                evaluate_gates(magical_convert, config, model, dict_, manager, ntokens, val_iters, device)

params = sum([np.prod(p.size()) for p in model.parameters()])
print("Model's params: %d" % params)

# Loop over epochs.
lr = config["lr"]
best_val_loss = None

if config["mode"] == "train" or config["mode"] == "finetune":
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, config["epochs"]+1):
            lr_ = lr * ((1.0/config["lr_decay"]) ** max(epoch - 10, 0.0))
            if "gate_lr" in config["diff_embedding"] and\
                (config["embedding"] == "diff_embedding" or config["embedding"] == "diff_embedding_continuous"):
                glr_ = config["diff_embedding"]["gate_lr"] *  ((1.0/config["lr_decay"]) ** max(epoch - 10, 0.0))
            else:
                glr_ = None
            epoch_start_time = time.time()

            for name, p in model.named_parameters():
                if "encoder.gates.weight" == name or "decoder.gates.weight" == name:
                    print(name, p.data[0:10])

            manager.train(
                model_type=config["model"],
                model=model,
                batch_size=config["batch_size"],
                train_iters=train_iters,
                val_iters=val_iters,
                epoch=epoch,
                lr=lr_,
                clip=config["clip"],
                log_interval=config["log_interval"],
                gate_clamping=config["diff_embedding"]["gate_clamping"]\
                    if "gate_clamping" in config["diff_embedding"] else None,
                gate_lr=glr_
            )
            val_loss = manager.evaluate(
                model_type=config["model"],
                model=model,
                data_source=val_iters,
                ntokens=ntokens,
                eval_batch_size=config["batch_size"])
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                val_loss = evaluate_gates(magical_convert, config, model, dict_, manager, ntokens, val_iters, device)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(config["save_path"], 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(config["save_path"], 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if config["model"] in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

# Run on test data.
test_loss = manager.evaluate(
    model_type=config["model"],
    model=model,
    data_source=test_iters,
    ntokens=ntokens,
    eval_batch_size=config["batch_size"])
val_loss = manager.evaluate(
    model_type=config["model"],
    model=model,
    data_source=val_iters,
    ntokens=ntokens,
    eval_batch_size=config["batch_size"])
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | val ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss), math.exp(val_loss)))
print('=' * 89)
