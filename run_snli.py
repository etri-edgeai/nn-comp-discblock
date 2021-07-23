import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)

import argparse
import yaml
import json

import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import logging


from tasks.snli.third_party import datasets
from tasks.snli.third_party import models
from tasks.snli.third_party.train import Train
from tasks.snli.third_party.evaluate import Evaluate

from tasks.snli.third_party.utils import *
from pdb import set_trace


from discblock import magic
from config_parser import get_embedding_options

def evaluate_gates(magical_convert, train_, model, args, config, device):

    load_func = lambda : load_model(
        args, train_.dataset.out_dim(), None, magical_convert, config["weights"], device=device
    )
    def eval_func(model_):
        return val_(train_, model_)
        
    val_score = magical_convert.evaluate_gates(model, config["block_options"], load_func, eval_func)
    print(val_score)
    return val_score

def val_(train_, model):
    omodel = train_.model
    train_.model = model
    val_acc = train_.validate()[1]
    train_.model = omodel
    return val_acc

def test_(eval_, model):
    eval_.model = model
    test_acc = eval_.execute()
    return test_acc

def train_caller(train_, model, config=None, evaluate_gates=None):
    train_.model = model
    train_.execute(config, evaluate_gates)

def load_model(args, out_dim, config=None, magical_convert=None, weight_path=None, device="cpu"):

    model_options = {
                    'out_dim': out_dim,
                    'dp_ratio': args.dp_ratio,
                    'd_hidden': args.d_hidden,
                    'device': args.gpu,
                    'dataset': args.dataset
    }
    model = models.__dict__[args.model](model_options)

    if weight_path is not None and weight_path != "none":
        model = torch.load(weight_path, map_location="cpu")["model"]
        model.to(device)

    if config is None:
        model.to(device)
        return model

    if weight_path is not None and weight_path != "none":
        if config["embedding"] != "none":
            magical_convert.convert(model, setup_weights=True)
    else:
        if config["embedding"] != "none":
            magical_convert.convert(model)

    model.to(device)
    return model


def run():

    train_ = Train()
    eval_ = Evaluate(train_.args, False)
    args = train_.args
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    options = get_embedding_options(config)
    device = torch.device(train_.model.device)
    print("+++++++ DEVICE:", device)

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
        model = torch.load(config["weights"], map_location="cpu")["model"]
        model.to(device)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)

        ret = test_(eval_, model)
        print("testing result: %f", ret)
    else:
        model = load_model(
            args,
            train_.dataset.out_dim(),
            config=config,
            magical_convert=magical_convert,
            weight_path=config["weights"],
            device=device)
        model.to(device)

        eval_func_ = None
        if config["weights"] is not None and config["weights"] != "none": 
            ret = test_(eval_, model)
            print("testing result: %f", ret)

            if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                eval_func_ = lambda model_: evaluate_gates(magical_convert, train_, model_, args, config, device)
                val_score = eval_func_(model)
                print("******************** eval initial: %f", val_score)

        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)

        train_caller(train_, model, config=config, evaluate_gates=eval_func_)

if __name__ == "__main__":
    run()
