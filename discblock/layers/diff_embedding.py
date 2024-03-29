""" Differentiable Embedding Base

Copyright 2022. ETRI all rights reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def discrete_mask(idx_array, gate):
    """ Discrete masking """
    return (idx_array < gate).to(dtype=torch.float32)

def get_mask(idx_array, gate, L=10e8, grad_shape_func=None):
    """ Masking function """
    if len(gate.size()) == 3:
        idx_array = idx_array.expand(gate.size()[0], gate.size()[1], -1).to(gate.device)
    elif len(gate.size()) == 2:
        idx_array = idx_array.expand(gate.size()[0], -1).to(gate.device)
    if callable(grad_shape_func):
        return discrete_mask(idx_array, gate) + ((L * gate - torch.floor(L * gate)) / L) * grad_shape_func(gate)
    else:
        return discrete_mask(idx_array, gate) + ((L * gate - torch.floor(L * gate)) / L)

def l2_reg_ortho_32bit(mdl):
    """ L2 Regularizer """

    l2_reg = None
    for name, W in mdl.named_parameters():
        if "gates_" in name:
            if W.ndimension() < 2:
                continue
            else:
                cols = W[0].numel()
                #rows = W.shape[0]
                #w1 = W.view(-1,cols)
                w1 = W
                wt = torch.transpose(w1,0,1)
                m  = torch.matmul(wt,w1)
                ident = Variable(torch.eye(cols,cols)).to(W)

                w_tmp = (m - ident)
                height = w_tmp.size(0)
                u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
                v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
                u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
                sigma = torch.dot(u, torch.matmul(w_tmp, v))
                if l2_reg is None:
                    l2_reg = (torch.norm(sigma,2))**2 / np.prod(w_tmp.size())
                    num = 1
                else:
                    l2_reg = l2_reg + ((torch.norm(sigma,2))**2 / np.prod(w_tmp.size()))
                    num += 1
    return l2_reg / num

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
            self.gates = nn.Embedding(int(vocab_size), 1, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding(int(vocab_size), int(output_dim))
            self.gates = nn.Embedding(int(vocab_size), 1)
        self.grad_shaping = grad_shape_func
        self.gates_blocks = nn.ModuleList([nn.Linear(output_dim, output_dim, bias=True)
        for i in range(5)])

        self.index_array = torch.FloatTensor([[[ i for i in range(output_dim)]]]).to(device)

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.reg_weight = reg_weight
        self._init_func = init_func

        self.padding_idx = padding_idx

        # Init
    def init_weights(self):
        """ Initialize """ 
        self._init_func(self.gates.weight.data, a=0.001, b=1.0)
        if self.padding_idx != -1:
            self.gates.weight.data[self.padding_idx] = 1.0
        self._init_func(self.embedding.weight.data)

        for b in self.gates_blocks:
            nn.init.eye_(b.weight)
            nn.init.zeros_(b.bias)

    def set_sparsity(self, sparsity):
        """ Set sparsity """
        self.sparsity = sparsity

    def get_sparsity(self, no_reduction=False):
        """ Get sparsity """
        if self.padding_idx != -1:
            floor_gates = torch.cat((self.gates.weight[:self.padding_idx], self.gates.weight[self.padding_idx+1:]))
        else:
            floor_gates = self.gates.weight
        if not no_reduction:
            return (1.0-floor_gates).mean()
        else:
            return (1.0-floor_gates)

    def get_sparsity_loss(self):
        """ Get sparsity loss """
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight

    def report(self):
        """ Report """
        print(self.get_sparsity())
    
    def forward(self, input):
        """ Forward """
        nobatch = False
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
            nobatch = True

        data = self.embedding(input)
        gates = self.gates(input) * self.embedding.weight.size()[1]
        mask = get_mask(self.index_array, gates, grad_shape_func=self.grad_shaping)

        bags = []
        for bidx, bm in enumerate(mask):
            vecs = []
            for widx, m in enumerate(bm):
                m_ = torch.sum(m)
                idx = int(m_ // (data.size()[2] // float(len(self.gates_blocks))))
                if idx == len(self.gates_blocks):
                    idx -= 1
                vecs.append(self.gates_blocks[idx](data[bidx][widx] * m))
            bags.append(torch.stack(vecs))

        if nobatch:
            return bags[0]
        else:
            return torch.stack(bags)

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

        self.gates = nn.Embedding(int(vocab_size), 1)
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, vocab_size))
        self.bias = nn.Parameter(torch.FloatTensor(vocab_size,))

        self.gates_blocks = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=True)
        for i in range(5)])

        # To compute gated array.
        self.index_array = torch.FloatTensor([[ i for i in range(input_dim)]]).to(device)

        self.grad_shaping = grad_shape_func
        self.sparsity = sparsity
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.reg_weight = reg_weight
        self._init_func = init_func

        # Init
    def init_weights(self):
        """ Initialize """
        self._init_func(self.gates.weight.data, a=0.001, b=1.0)
        self._init_func(self.weight.data)
        self._init_func(self.bias.data)

        for b in self.gates_blocks:
            nn.init.eye_(b.weight)
            nn.init.zeros_(b.bias)

    def set_sparsity(self, sparsity):
        """ Set Sparsity """
        self.sparsity = sparsity

    def get_sparsity(self, no_reduction=False):
        """ Get sparsity """
        floor_gates = self.gates.weight
        if not no_reduction:
            return (1.0-floor_gates).mean()
        else:
            return (1.0-floor_gates)

    def get_sparsity_loss(self):
        """ Get sparsity loss """
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight

    def report(self):
        """ Report """
        print(self.get_sparsity())
    
    def forward(self, input):
        """ Forward """

        all_ = torch.arange(self.vocab_size).to(input.device)
        gates = self.gates(all_) * self.weight.size()[0]
        mask_ = get_mask(self.index_array, gates, grad_shape_func=self.grad_shaping)
        mask = torch.transpose(mask_, 0, 1)
        masked_weight = self.weight * mask

        selector = (gates / input.size()[-1]) * len(self.gates_blocks)
        selector = torch.floor(selector * (1 - 1e-10))

        ret = None
        for bidx, b in enumerate(self.gates_blocks):
            input_ = b(input)
            temp = torch.matmul(input_, masked_weight)

            selector_ = (selector == bidx).float().view((self.vocab_size,))

            if ret is None:
                ret = temp * selector_
            else:
                ret += temp * selector_
        return ret  + self.bias
