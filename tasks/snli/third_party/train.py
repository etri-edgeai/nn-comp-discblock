import os

import time
import datetime

import torch
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import logging


from tasks.snli.third_party import datasets
from tasks.snli.third_party import models

from tasks.snli.third_party.utils import *
from pdb import set_trace

class Train():
        def __init__(self):
                print("program execution start: {}".format(datetime.datetime.now()))
                self.args = parse_args()
                self.device = get_device(self.args.gpu)
                self.logger = get_logger(self.args, "train")
                self.logger.info("Arguments: {}".format(self.args))
                
                dataset_options = {
                                                                                        'batch_size': self.args.batch_size, 
                                                                                        'device': self.device
                                                                                }
                print(datasets.__dict__.keys())
                self.dataset = datasets.__dict__[self.args.dataset](dataset_options)
                
                self.model_options = {
                                                                        'out_dim': self.dataset.out_dim(),
                                                                        'dp_ratio': self.args.dp_ratio,
                                                                        'd_hidden': self.args.d_hidden,
                                                                        'device': self.device,
                                                                        'dataset': self.args.dataset
                                                                }
                self.model = models.__dict__[self.args.model](self.model_options)
                self.model.to(self.device)
                print("resource preparation done: {}".format(datetime.datetime.now()))

        def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
                if self.best_val_acc is None or val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        torch.save({
                                'accuracy': self.best_val_acc,
                                'options': self.model_options,
                                'model_dict': self.model.state_dict(),
                        }, '{}/{}/{}/best-{}-{}-params.pt'.format(self.args.results_dir, self.args.model, self.args.dataset, self.args.model, self.args.dataset))
                        torch.save({
                                'accuracy': self.best_val_acc,
                                'options': self.model_options,
                                'model': self.model,
                        }, '{}/{}/{}/best-{}-{}-params.pth'.format(self.args.results_dir, self.args.model, self.args.dataset, self.args.model, self.args.dataset))
                self.logger.info('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'
                                .format(epoch, train_loss, train_acc, val_loss, val_acc, took))
        
        def train(self, config=None):

                self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
                if config is not None and "diff_embedding" in config["embedding"] and "gate_lr" in config["diff_embedding"]:
                    if config["diff_embedding"]["gate_training_only"]:
                        print("USE GATE LR:", config["diff_embedding"]["gate_lr"])
                        lr = config["diff_embedding"]["gate_lr"]
                    else:
                        lr = self.args.lr
                else:
                    lr = self.args.lr

                self.opt = O.Adam([p for p in self.model.parameters() if p.requires_grad], lr = lr)
                self.best_val_acc = None
                self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)

                self.model.train(); self.dataset.train_iter.init_epoch()
                n_correct, n_total, n_loss = 0, 0, 0
                for batch_idx, batch in enumerate(self.dataset.train_iter):
                        self.opt.zero_grad()
                        answer = self.model(batch)
                        loss = self.criterion(answer, batch.label)
                        
                        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                        n_total += batch.batch_size
                        n_loss += loss.item()

                        sparsity_loss = 0.0
                        for c in self.model.children():
                            if type(c).__name__ == "DifferentiableEmbedding" or type(c).__name__ == "DifferentiableEmbeddingClassifier":
                                sparsity_loss += c.get_sparsity_loss()

                        (loss+sparsity_loss).backward(); self.opt.step()

                        if config is not None and "diff_embedding" in config["embedding"] and "gate_clamping" in config["diff_embedding"]:
                            gate_clamping = config["diff_embedding"]["gate_clamping"]
                            for c in self.model.modules():
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

                train_loss = n_loss/n_total
                train_acc = 100. * n_correct/n_total
                return train_loss, train_acc

        def validate(self):
                if not hasattr(self, "criterion"):
                    self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
                self.model.eval(); self.dataset.dev_iter.init_epoch()
                n_correct, n_total, n_loss = 0, 0, 0
                with torch.no_grad():
                        for batch_idx, batch in enumerate(self.dataset.dev_iter):
                                answer = self.model(batch)
                                loss = self.criterion(answer, batch.label)
                                
                                n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                                n_total += batch.batch_size
                                n_loss += loss.item()

                        val_loss = n_loss/n_total
                        val_acc = 100. * n_correct/n_total
                        return val_loss, val_acc

        def execute(self, config=None, evaluate_gates=None):
                print(" [*] Training starts!")
                print('-' * 99)
                for epoch in range(1, self.args.epochs+1):
                        start = time.time()

                        train_loss, train_acc = self.train(config)
                        val_loss, val_acc = self.validate()
                        self.scheduler.step()
                        
                        if config is not None and "use_eval_gates" in config and config["use_eval_gates"] and "diff_embedding" in config["embedding"]:
                            val_acc = evaluate_gates(self.model)
                            print("********************* eval gates: %f", val_acc)

                        took = time.time()-start
                        self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

                        print('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'.format(
                                epoch, train_loss, train_acc, val_loss, val_acc, took))
                self.finish()

        def finish(self):
                self.logger.info("[*] Training finished!\n\n")
                print('-' * 99)
                print(" [*] Training finished!")
                print(" [*] Please find the saved model and training log in results_dir")

#task = Train()
#task.execute()
