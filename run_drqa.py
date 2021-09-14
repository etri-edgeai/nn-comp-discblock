#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main DrQA reader training script."""

import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)


import yaml
import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging
import inspect

from drqa.reader import utils, vector, config, data
from drqa.reader import DocReader
from drqa import DATA_DIR as DRQA_DATA

from discblock import magic
from discblock.utils import compute_sparsity_loss
from config_parser import get_embedding_options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = '/tmp/drqa-models/'
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')

def evaluate_gates(magical_convert, model, args, argconfig, feature_dict, word_dict, dev_loader, stats, dev_offsets, dev_texts, dev_answers):
    saved_params = torch.load(
        argconfig["weights"], map_location=lambda storage, loc: storage
    )
    word_dict = saved_params['word_dict']
    feature_dict = saved_params['feature_dict']
    state_dict = saved_params['state_dict']
    model_ = DocReader(args, word_dict, feature_dict, state_dict, normalize=True)
    if args.embedding_file:
        model_.load_embeddings(word_dict.tokens(), args.embedding_file)
    if args.cuda:
        model_.cuda()
    def load_func():
        return model_.network

    def eval_func(network):
        model_.network = network.to(torch.device(argconfig["device"]))

        stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
        #validate_unofficial(args, train_loader, model, stats, mode='train')
        result = validate_unofficial(args, dev_loader, model_, stats, mode='dev')
        result = validate_official(args, dev_loader, model_, stats,
                                   dev_offsets, dev_texts, dev_answers)
        return result
        
    val_score = magical_convert.evaluate_gates(model.network, argconfig["block_options"], load_func, eval_func)
    return val_score

def load_model(args, argconfig, magical_convert, train_exs, dev_exs, weight_path=None, direct_load=False):

    if weight_path is not None and weight_path != "none":
        saved_params = torch.load(
            weight_path, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize=True)
        if direct_load:
            return model
    else:
        # Construct base model
        feature_dict = utils.build_feature_dict(args, train_exs)
        word_dict = utils.build_word_dict(args, train_exs + dev_exs)
        model = DocReader(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    if weight_path is not None and weight_path != "none":
        if argconfig["embedding"] != "none":
            magical_convert.convert(model.network, setup_weights=True)
    else:
        if argconfig["embedding"] != "none":
            magical_convert.convert(model.network)
    return model, word_dict, feature_dict

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=16,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-processed-corenlp.txt',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-corenlp.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--config', type=str, default=None, help='configuration', required=True)


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError('No such file: %s' % args.dev_json)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.best_model_file = os.path.join(args.model_dir, 'best.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    args.tune_partial = 0
    args.fix_embeddings = False

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, argconfig):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # check sparsity_loss_func
    if "sparsity_loss_func" not in inspect.getfullargspec(model.update)[0]:
        logger.warning("DRQA is not changed to handle sparsity loss for differentiable scoring.") 

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex, sparsity_loss_func=compute_sparsity_loss))

        if "diff_embedding" in argconfig["embedding"] and "gate_clamping" in argconfig["diff_embedding"]:
            gate_clamping = argconfig["diff_embedding"]["gate_clamping"]
            for c in model.network.modules():
                if type(c).__name__ == "DifferentiableEmbedding":
                    #torch.clamp_(c.gates.weight.data, min=1.0, max=c.embedding.weight.size()[1])
                    if type(c.gates).__name__ == "Embedding":
                        torch.clamp_(c.gates.weight.data, min=gate_clamping[0], max=gate_clamping[1])
                    else:
                        torch.clamp_(c.gates.data, min=gate_clamping[0], max=gate_clamping[1])
                elif type(c).__name__ == "DifferentiableEmbeddingClassifier":
                    #torch.clamp_(c.gates.weight.data, min=1.0, max=c.weight.size()[0])
                    if type(c.gates).__name__ == "Embedding":
                        torch.clamp_(c.gates.weight.data, min=gate_clamping[0], max=gate_clamping[1])
                    else:
                        torch.clamp_(c.gates.data, min=gate_clamping[0], max=gate_clamping[1])

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate_unofficial(args, data_loader, model, global_stats, mode):
    """Run one full unofficial validation.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = utils.Timer()
    start_acc = utils.AverageMeter()
    end_acc = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)
        target_s, target_e = ex[-3:-1]

        # We get metrics for independent start/end and joint start/end
        accuracies = eval_accuracies(pred_s, target_s, pred_e, target_e)
        start_acc.update(accuracies[0], batch_size)
        end_acc.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial: Epoch = %d | start = %.2f | ' %
                (mode, global_stats['epoch'], start_acc.avg) +
                'end = %.2f | exact = %.2f | examples = %d | ' %
                (end_acc.avg, exact_match.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'exact_match': exact_match.avg}


def validate_official(args, data_loader, model, global_stats,
                      offsets, texts, answers):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.

    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)

        for i in range(batch_size):
            s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
            e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
            prediction = texts[ex_id[i]][s_offset:e_offset]

            # Compute metrics
            ground_truths = answers[ex_id[i]]
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))

        examples += batch_size

    logger.info('dev valid official: Epoch = %d | EM = %.2f | ' %
                (global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    """An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e.item()] for e in target_s]
        target_e = [[e.item()] for e in target_e]

    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args, argconfig):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # If we are doing offician evals then we need to:
    # 1) Load the original text to retrieve spans from offsets.
    # 2) Load the (multiple) text answers for each question.
    if args.official_eval:
        dev_texts = utils.load_text(args.dev_json)
        dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
        dev_answers = utils.load_answers(args.dev_json)

    options = get_embedding_options(argconfig)

    device = torch.device(argconfig["device"])
    magical_convert = magic.EmbeddingMagic(
    mode=argconfig["mode"],
    embeddings=argconfig["embeddings"],
    target_ratio=argconfig["target_ratio"],
    embedding_type=argconfig["embedding"],
    options=options,
    use_embedding_for_decoder=argconfig["use_embedding_for_decoder"],
    device=device)


    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if argconfig["mode"] == "test":
        model = load_model(args, argconfig, magical_convert, train_exs, dev_exs, argconfig["weights"], direct_load=True)
    else:
       
        if "lr_mm" in argconfig:
            args.learning_rate = args.learning_rate * argconfig["lr_mm"]
            print("NEW LR:", args.learning_rate)

        if "diff_embedding" in argconfig["embedding"] and argconfig["diff_embedding"]["gate_training_only"]:
            args.learning_rate = argconfig["diff_embedding"]["gate_lr"]

        model, word_dict, feature_dict = load_model(args, argconfig, magical_convert, train_exs, dev_exs, argconfig["weights"], direct_load=False)

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    f1_scores = []
    exact_match_scores = []

    # Log model parameter status
    # quant_embed.log_param_list(model.network)

    if argconfig["mode"] == "test":
        stats['epoch'] = 0

        # Validate unofficial (train)
        # validate_unofficial(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        f1_scores.append(result['f1'])
        exact_match_scores.append(result['exact_match'])
        return f1_scores, exact_match_scores

    if "use_eval_gates" in argconfig and argconfig["use_eval_gates"] and "diff_embedding" in argconfig["embedding"]:
        result = evaluate_gates(magical_convert, model, args, argconfig, feature_dict, word_dict, dev_loader, stats, dev_offsets, dev_texts, dev_answers)
        print("gates: ", result)

    if argconfig["embedding"] != "none":

        # Validate unofficial (dev)
        result = validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)
        print(result)

    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats, argconfig)

        # Validate unofficial (train)
        # validate_unofficial(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        if "use_eval_gates" in argconfig and argconfig["use_eval_gates"] and "diff_embedding" in argconfig["embedding"]:
            result = evaluate_gates(magical_convert, model, args, argconfig, feature_dict, word_dict, dev_loader, stats, dev_offsets, dev_texts, dev_answers)
            print("gates: ", result)

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.best_model_file)
            stats['best_valid'] = result[args.valid_metric]

        f1_scores.append(result['f1'])
        exact_match_scores.append(result['exact_match'])

    return f1_scores,exact_match_scores

def train_drqa(cmdline_args, use_cuda=True):
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args(cmdline_args)
    set_defaults(args)

    # apply config.
    with open(args.config, 'r') as stream:
        try:
            argconfig = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    torch.cuda.set_device(argconfig["device"])
    # Set cuda
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.cuda = use_cuda and torch.cuda.is_available()

    # NOTE: The code below is commented out because it is done in smallfry utils.
    # Set random state
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.random_seed)

    # Set logging
    # logger.setLevel(logging.INFO)
    # fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
    #                         '%m/%d/%Y %I:%M:%S %p')
    # console = logging.StreamHandler()
    # console.setFormatter(fmt)
    # logger.addHandler(console)
    # if args.log_file:
    #     if args.checkpoint:
    #         logfile = logging.FileHandler(args.log_file, 'a')
    #     else:
    #         logfile = logging.FileHandler(args.log_file, 'w')
    #     logfile.setFormatter(fmt)
    #     logger.addHandler(logfile)
    # logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    return main(args, argconfig)

if __name__ == '__main__':

    print(sys.argv[1:])

    train_drqa(sys.argv[1:])
