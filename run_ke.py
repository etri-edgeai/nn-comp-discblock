# coding: utf-8
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

from discblock import magic
from config_parser import get_embedding_options

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, RotatE, TransD
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser(description='', add_help=False)
parser.add_argument('--config', type=str, default=None, help='configuration')
args = parser.parse_args()

with open(args.config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

if config["dataset"] == "fb15k237":

    # dataloader for training
    train_dataloader = TrainDataLoader(
            in_path = "../../../tasks/ke/third_party/benchmarks/FB15K237/",
            nbatches = 100,
            threads = 8,
            sampling_mode = "normal",
            bern_flag = 1,
            filter_flag = 1,
            neg_ent = 25,
            neg_rel = 0)

    # dataloader for test
    test_dataloader = TestDataLoader("../../../tasks/ke/third_party/benchmarks/FB15K237/", "link")

    lr = 1.0
    opt_method = None

else:
    raise NotImplementedError()

def train_(model, epochs=10, alpha=1.0, config=None, eval_func=None):
    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = epochs, alpha = alpha, use_gpu = True, opt_method=opt_method, config=config, eval_gates=eval_func)
    trainer.run()

def test_(model):
    tester = Tester(model = model, data_loader = test_dataloader, use_gpu = True)
    return tester.run_link_prediction(type_constrain = False)

def evaluate_gates(magical_convert, config, model, device):

    load_func = lambda : load_model(
        None, magical_convert, config["weights"], device=device, dataset=config["dataset"], model_name=config["model"]
    )
    def eval_func(model_):
        return test_(model_.model)
 
    val_score = magical_convert.evaluate_gates(model, config["block_options"], load_func, eval_func)
    print(val_score)
    return val_score

def load_model(config=None, magical_convert=None, weight_path=None, device="cpu", dataset="fb15k237", model_name="transe"):

    if dataset == "fb15k237":

        if model_name == "transe":
            # define the model
            transe = TransE(
                    ent_tot = train_dataloader.get_ent_tot(),
                    rel_tot = train_dataloader.get_rel_tot(),
                    dim = 200,
                    p_norm = 1,
                    norm_flag = True)

            # define the loss function
            model = NegativeSampling(
                    model = transe,
                    loss = MarginLoss(margin = 5.0),
                    batch_size = train_dataloader.get_batch_size()
            )
        elif model_name == "transd":

            # define the model
            transd = TransD(
                    ent_tot = train_dataloader.get_ent_tot(),
                    rel_tot = train_dataloader.get_rel_tot(),
                    dim_e = 200,
                    dim_r = 200,
                    p_norm = 1,
                    norm_flag = True)


            # define the loss function
            model = NegativeSampling(
                    model = transd,
                    loss = MarginLoss(margin = 4.0),
                    batch_size = train_dataloader.get_batch_size()
            )

        elif model_name == "rotate":
            # define the model
            rotate = RotatE(
                    ent_tot = train_dataloader.get_ent_tot(),
                    rel_tot = train_dataloader.get_rel_tot(),
                    dim = 1024,
                    margin = 4.0,
                    epsilon = 2.0,
            )

            # define the loss function
            model = NegativeSampling(
                    model = rotate,
                    loss = MarginLoss(margin = 4.0),
                    batch_size = train_dataloader.get_batch_size()
            )

        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    print(model)

    if weight_path is not None and weight_path != "none":
        model = torch.load(weight_path, map_location="cpu")
        model.to(device)

    if config is None:
        model.to(device)
        return model
    else:
        if "diff_embedding" in config["embedding"]:
            torch.nn.init.constant_(model.model.ent_embeddings.gates.weight.data, 0.5)

    if weight_path is not None and weight_path != "none":
        if config["embedding"] != "none":
            magical_convert.convert(model, setup_weights=True)
    else:
        if config["embedding"] != "none":
            magical_convert.convert(model)

    model.to(device)
    return model


# Set the random seed manually for reproducibility.
torch.manual_seed(config["seed"])
device = torch.device(config["device"])

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
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Model's params: %d" % params)

    ret = test_(model.model)
    print("testing result: %f", ret)

else:

    model = load_model(config, magical_convert, config["weights"], device, config["dataset"], config["model"])
    model.to(device)

    eval_func_ = None
    if config["weights"] is not None and config["weights"] != "none": 
        ret = test_(model.model)
        print("testing result: %f", ret)
        if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
            eval_func_ = lambda model_: evaluate_gates(magical_convert, config, model_, device)
            val_score = eval_func_(model)
            print("******************** eval initial: %f", val_score)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Model's params: %d" % params)

    model.batch_size = train_dataloader.get_batch_size()
    train_(model, config["epochs"], config["lr_mm"] * lr, config=config, eval_func=eval_func_)
    with open(config["save_path"], 'wb') as f:
        torch.save(model, f)
