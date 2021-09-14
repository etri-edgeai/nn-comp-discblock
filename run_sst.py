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

from tasks.sst.third_party import models
from tasks.sst.third_party.config import Config
from tasks.sst.third_party.models.TextCNN import ModelCNN
from tasks.sst.third_party.models.TextAttnBiLSTM import ModelAttnBiLSTM
from tasks.sst.third_party.datasets import *
from tasks.sst.third_party.utils import *

from discblock import magic
from config_parser import get_embedding_options


def evaluate_gates(magical_convert, model, opt, argconfig, val_loader):

    device = argconfig["device"]
    criterion = nn.CrossEntropyLoss().to(device)
    load_func = lambda : load_model(
        opt, argconfig, magical_convert, argconfig["weights"], direct_load=True, device=device
    )

    def eval_func(model_):
        return val_(val_loader, criterion, opt, model_, device=device)
        
    val_score = magical_convert.evaluate_gates(model, argconfig["block_options"], load_func, eval_func)
    print(val_score)
    return val_score

def load_model(opt, argconfig, magical_convert, weight_path=None, direct_load=False, device="cpu"):

    word_map_file = opt.output_folder +  opt.data_name + '_' + 'wordmap.json'
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    embed_file = opt.output_folder + opt.data_name + '_' + 'pretrain_embed.pth'
    embed_file = torch.load(embed_file)
    pretrain_embed, embed_dim = embed_file['pretrain'], embed_file['dim']

    if weight_path is not None and weight_path != "none":
        model = torch.load(weight_path, map_location='cpu')["model"]
    else:
        if opt.model_name == "TextCNN":
            model = ModelCNN(vocab_size=len(word_map), 
                          embed_dim=embed_dim, 
                          kernel_num=opt.kernel_num, 
                          kernel_sizes=opt.kernel_sizes, 
                          class_num=opt.class_num,
                          pretrain_embed=pretrain_embed,
                          dropout=opt.dropout, 
                          static=opt.static, 
                          non_static=opt.non_static, 
                          multichannel=opt.multichannel)
        else:
            # Construct a base model 
            model = ModelAttnBiLSTM(vocab_size=len(word_map), 
                          embed_dim=embed_dim, 
                          hidden_size=embed_dim,
                          class_num=opt.class_num,
                          pretrain_embed=pretrain_embed,
                          num_layers=opt.num_layers,
                          model_dropout=opt.model_dropout, 
                          fc_dropout=opt.fc_dropout,
                          embed_dropout=opt.embed_dropout,
                          use_gru=opt.use_gru, 
                          use_embed=opt.use_embed)

    model = model.to(device)
    if direct_load:
        return model

    if weight_path is not None and weight_path != "none":
        if argconfig["embedding"] != "none":
            magical_convert.convert(model, setup_weights=True)
    else:
        if argconfig["embedding"] != "none":
            magical_convert.convert(model)

    if "diff_embedding" in argconfig["embedding"] and argconfig["diff_embedding"]["gate_training_only"]:
        lr = argconfig["diff_embedding"]["gate_lr"]
    elif "lr_mm" in argconfig:
        lr = argconfig["lr_mm"] * opt.lr
        print("NEW LR:", lr)
    else:
        lr = opt.lr

    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                                 lr=lr,
                                 weight_decay=opt.weight_decay)

    return model, optimizer, word_map

def val_(val_loader, criterion, opt, model, device):
    acc = validate(val_loader=val_loader,
                  model=model,
                  criterion=criterion,
                  print_freq=opt.print_freq,
                  device=device)
    return acc

def test_(opt, model, device="cpu"):

    # 移动到GPU
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        SSTreebankDataset(opt.data_name, opt.output_folder, 'test'),
        batch_size=opt.batch_size, 
        shuffle=False,
        num_workers = opt.workers if opt.is_Linux else 0,
        pin_memory=True)
    
    # test
    ret = testing(test_loader, model, criterion, opt.print_freq, device)
    return ret

def train_(opt, model, optimizer, config, magical_convert, word_map, device="cpu"):

    # 初始化best accuracy
    best_acc = 0.

    # epoch
    start_epoch = 0
    epochs = opt.epochs
    epochs_since_improvement = 0  # 跟踪训练时的验证集上的BLEU变化，每过一个epoch没提升则加1

    # loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'train'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
                    SSTreebankDataset(opt.data_name, opt.output_folder, 'dev'),
                    batch_size=opt.batch_size, 
                    shuffle=True,
                    num_workers = opt.workers if opt.is_Linux else 0,
                    pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        
        # 学习率衰减
        if epoch > opt.decay_epoch:
            adjust_learning_rate(optimizer, epoch)
        
        # early stopping 如果dev上的acc在6个连续epoch上没有提升
        if epochs_since_improvement == opt.improvement_epoch:
            break
       
        if hasattr(opt, "grad_clip"):
            grad_clip = opt.grad_clip
        else:
            grad_clip = None

        # 一个epoch的训练
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              vocab_size=len(word_map),
              print_freq=opt.print_freq,
              device=device,
              grad_clip=grad_clip,
              config=config)
        
        # 一个epoch的验证
        recent_acc = validate(val_loader=val_loader,
                              model=model,
                              criterion=criterion,
                              print_freq=opt.print_freq,
                              device=device)

        if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
            recent_acc = evaluate_gates(magical_convert, model, opt, config, val_loader)
            print("********************* eval gates: %f", recent_acc)
        
        # 检查是否有提升
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
        
        # 保存模型
        save_checkpoint(opt.model_name, opt.data_name, epoch, epochs_since_improvement, model, optimizer, recent_acc, is_best)

    print(test_(opt, model, device))


def run():

    parser = argparse.ArgumentParser(description='', add_help=False)
    parser.add_argument('--config', type=str, default=None, help='configuration')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    global_opt = Config()

    if global_opt.use_model == 'TextCNN':
        model_opt = models.TextCNN.ModelConfig()
    elif global_opt.use_model == 'TextAttnBiLSTM':
        model_opt = models.TextAttnBiLSTM.ModelConfig()  
    else:
        raise NotImplementedError("Not supported model: %s", model_opt)

    options = get_embedding_options(config)
    device = config["device"]

    magical_convert = magic.EmbeddingMagic(
        mode=config["mode"],
        embeddings=config["embeddings"],
        target_ratio=config["target_ratio"],
        embedding_type=config["embedding"],
        options=options,
        use_embedding_for_decoder=config["use_embedding_for_decoder"],
        device=device)

    # load model
    if "test" in config["mode"]:
        if "_init" in config["mode"]:
            model, optimizer, word_map = load_model(model_opt, config, magical_convert, weight_path=config["weights"], direct_load=False)
        else:
            model = torch.load(config["weights"], map_location="cpu")["model"]
        model.to(device)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)
        print(model)

        ret = test_(model_opt, model, device)
        print("testing result: %f", ret)
    else:
        model, optimizer, word_map = load_model(model_opt, config, magical_convert, weight_path=config["weights"], direct_load=False)
        model.to(device)

        if config["weights"] is not None and config["weights"] != "none": 
            ret = test_(model_opt, model, device)
            print("testing result: %f", ret)

            if "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                opt = model_opt
                val_loader = torch.utils.data.DataLoader(
                                SSTreebankDataset(opt.data_name, opt.output_folder, 'dev'),
                                batch_size=opt.batch_size, 
                                shuffle=True,
                                num_workers = opt.workers if opt.is_Linux else 0,
                                pin_memory=True)

                val_score = evaluate_gates(magical_convert, model, model_opt, config, val_loader)
                print("******************** eval initial: %f", val_score)

        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Model's params: %d" % params)

        train_(model_opt, model, optimizer, config, magical_convert, word_map, device=device)

if __name__ == "__main__":
    run()
