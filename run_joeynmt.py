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
import argparse
import time
import shutil
from typing import List
import logging
import os
import sys
import collections
import pathlib

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Dataset

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.training import TrainManager
from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, latest_checkpoint_update, \
    ConfigurationError
from joeynmt.model import Model, _DataParallel
from joeynmt.prediction import validate_on_data, parse_test_args
from joeynmt.loss import XentLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper

# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    # error handling in TrainManager object construction
    pass

logger = logging.getLogger(__name__)

from bec import magic
from config_parser import get_embedding_options


def evaluate_gates(magical_convert, model, trainer, cfg, src_vocab, trg_vocab, config, valid_data):

    device = torch.device("cuda")
    load_func = lambda : load_model(
        cfg, src_vocab, trg_vocab, None, magical_convert, config["weights"], device=device
    )
    def eval_func(model_):
        return validate(trainer, model_, valid_data)
        
    val_score, val_duration = magical_convert.evaluate_gates(model, config["block_options"], load_func, eval_func)
    return val_score, val_duration


def validate(trainer, model, valid_data):
    return trainer._validate(valid_data, -1, model=model, ret_score=True)


def test(cfg, model, datasets=None):
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param batch_class: class type of batch
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """

    model_dir = cfg["training"]["model_dir"]
    batch_class = Batch

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # load the data
    if datasets is None:
        _, dev_data, test_data, src_vocab, trg_vocab = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        logger.info("Decoding on %s set (%s)...", data_set_name, dataset_file)

        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=data_set, batch_size=batch_size,
            batch_class=batch_class, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric=eval_metric,
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu)
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            logger.info("%4s %s%s: %6.2f [%s]",
                        data_set_name, eval_metric, tokenizer_info,
                        score, decoding_description)
            print(data_set_name, eval_metric, tokenizer_info, score, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)


def train(cfg, config, trainer, model_dir, model, train_data, dev_data, test_data, src_vocab, trg_vocab):
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """

    # store copy of original training config in model dir
    # shutil.copy2(cfg_file, model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg)

    log_data_info(train_data=train_data,
                  valid_data=dev_data,
                  test_data=test_data,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab)

    logger.info(str(model))

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(model_dir, trainer.stats.best_ckpt_iter)
    output_name = "{:08d}.hyps".format(trainer.stats.best_ckpt_iter)
    output_path = os.path.join(model_dir, output_name)
    datasets_to_test = {
        "dev": dev_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }
    test(cfg,
         model,
         datasets=datasets_to_test)


def load_model(cfg, src_vocab, trg_vocab, config=None, magical_convert=None, weight_path=None, device="cpu"):

    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    if weight_path is not None and weight_path != "none":
        model_state = torch.load(weight_path, map_location="cpu")["model_state"]
        model.load_state_dict(model_state)

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

    ap = argparse.ArgumentParser("Joey NMT")
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")
    ap.add_argument('--config', type=str, default=None, help='configuration')

    args = ap.parse_args()
    cfg = load_config(args.config_path)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])

    datasets_to_test = {
        "dev": dev_data,
        "test": test_data,
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab
    }

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    options = get_embedding_options(config)
    device = torch.device("cuda")
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

        model = load_model(cfg, src_vocab, trg_vocab, config, magical_convert, weight_path=config["weights"], device=device)
        model.to(device)

        if "weights2" in config and "init" not in config["weights2"]:
            m = torch.load(config["weights2"], map_location="cpu")["model_state"]
            model.load_state_dict(m)

        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)

        ret = test(cfg, model, datasets=datasets_to_test)
        print("testing result: %f", ret)
    else:

        model = load_model(cfg, src_vocab, trg_vocab, config, magical_convert, weight_path=config["weights"], device=device)
        model.to(device)

        # make logger
        model_dir = make_model_dir(cfg["training"]["model_dir"],
                                   overwrite=cfg["training"].get(
                                       "overwrite", False))
        _ = make_logger(model_dir, mode="train")  # version string returned
        # TODO: save version number in model checkpoints

        # set the random seed
        set_seed(seed=cfg["training"].get("random_seed", 42))

        if "epochs" in config:
            cfg["training"]["epochs"] = config["epochs"]

        trainer = TrainManager(model=model, config=cfg, argconfig=config)

        eval_func_ = None
        if config["weights"] is not None and config["weights"] != "none": 
            ret = test(cfg, model, datasets=datasets_to_test)
            print("testing result: %f", ret)
            if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                eval_func_ = lambda model_: evaluate_gates(magical_convert, model_, trainer, cfg, src_vocab, trg_vocab, config, valid_data=dev_data)
                trainer.eval_func = eval_func_
                val_score = eval_func_(model)[0]
                print("******************** eval initial: %f", val_score)

        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)
        train(cfg, config, trainer, model_dir, model, train_data, dev_data, test_data, src_vocab, trg_vocab)

if __name__ == "__main__":
    run()
