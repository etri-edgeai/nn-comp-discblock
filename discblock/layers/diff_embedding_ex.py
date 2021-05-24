from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch

def b(x):
    # Assume that > operator must be supported in backends.
    return (x >= 0.5).to(torch.float32)

def gate_func(x, L=10e5, grad_shape_func=None):
    x_ = x - 0.5
    if callable(grad_shape_func):
        return b(x) + ((L * x_ - torch.floor(L * x_)) / L) * grad_shape_func(x_)
    else:
        return b(x) + ((L * x_ - torch.floor(L * x_)) / L)

class DifferentiableEmbedding(nn.Module):
    """Differentiable Embedding Implementation

    """
    def __init__(self,
                 vocab_size,
                 output_dim,
                 sparsity=None,
                 grad_shape_func=None,
                 init_func=nn.init.uniform_,
                 reg_weight=1.0,
                 device="cuda:0",
                 padding_idx=-1):
        super(DifferentiableEmbedding, self).__init__()

        if padding_idx != -1:
            self.embedding = nn.Embedding(int(vocab_size), int(output_dim), padding_idx=padding_idx)
            self.gates = nn.Embedding(int(vocab_size), int(output_dim), padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(int(vocab_size), int(output_dim))
            self.gates = nn.Embedding(int(vocab_size), int(output_dim))

        self.grad_shaping = grad_shape_func

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.reg_weight = reg_weight
        self._init_func = init_func

        self.padding_idx = padding_idx

        # Init
    def init_weights(self):
        self._init_func(self.gates.weight.data, a=0.001, b=1.0)
        if self.padding_idx != -1:
            self.gates.weight.data[self.padding_idx] = 1.0
        self._init_func(self.embedding.weight.data)

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def binary_selection(self):
        return b(self.gates.weight)

    def diff_selection(self, gates=None):
        if gates is None:
            return gate_func(self.gates.weight)
        else:
            return gate_func(gates)

    def partial_diff_selection(self, gates):
        return gate_func(gates)

    def get_sparsity(self, no_reduction=False):
        if self.padding_idx != -1:
            floor_gates = torch.cat((self.gates.weight[:self.padding_idx], self.gates.weight[self.padding_idx+1:]))
        else:
            floor_gates = self.gates.weight

        selection = self.diff_selection(floor_gates)
        if not no_reduction:
            return 1.0 - (torch.sum(selection) / np.prod(floor_gates.size())).mean()
        else:
            return 1.0 - (torch.sum(selection) / np.prod(floor_gates.size()))

    def get_sparsity_loss(self):
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight

    def report(self):
        print(self.get_sparsity())
    
    def forward(self, input):
        data = self.embedding(input)
        gates = self.partial_diff_selection(self.gates(input))
        return data * gates

class DifferentiableEmbeddingClassifier(nn.Module):
    """Differentiable Embedding Implementation

    """
    def __init__(self,
                 vocab_size,
                 input_dim,
                 sparsity=None,
                 grad_shape_func=None,
                 init_func=nn.init.uniform_,
                 reg_weight=1.0,
                 device="cuda:0"):
        super(DifferentiableEmbeddingClassifier, self).__init__()

        self.gates = nn.Embedding(int(vocab_size), input_dim)
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, vocab_size))
        self.bias = nn.Parameter(torch.FloatTensor(vocab_size,))

        self.grad_shaping = grad_shape_func
        self.sparsity = sparsity
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.reg_weight = reg_weight
        self._init_func = init_func

        # Init
    def init_weights(self):
        self._init_func(self.gates.weight.data, a=0.001, b=1.0)
        self._init_func(self.weight.data)
        self._init_func(self.bias.data)

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def binary_selection(self):
        return b(self.gates.weight)

    def diff_selection(self):
        return gate_func(self.gates.weight)

    def get_sparsity(self, no_reduction=False):
        selection = self.diff_selection()
        if not no_reduction:
            return 1.0 - (torch.sum(selection) / np.prod(self.gates.weight.size())).mean()
        else:
            return 1.0 - (torch.sum(selection) / np.prod(self.gates.weight.size()))

    def get_sparsity_loss(self):
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight

    def report(self):
        print(self.get_sparsity())
    
    def forward(self, input):
        selection = self.diff_selection()
        masked_weight = self.weight * torch.transpose(selection, 0, 1)
        return torch.matmul(input, masked_weight) + self.bias
