from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def discrete_mask(idx_array, gate):
    return (idx_array < gate).to(dtype=torch.float32)

def get_mask(idx_array, gate, L=10e8, grad_shape_func=None):
    if len(gate.size()) == 3:
        idx_array = idx_array.expand(gate.size()[0], gate.size()[1], -1).to(gate.device)
    elif len(gate.size()) == 2:
        idx_array = idx_array.expand(gate.size()[0], -1).to(gate.device)
    if callable(grad_shape_func):
        return discrete_mask(idx_array, gate) + ((L * gate - torch.floor(L * gate)) / L) * grad_shape_func(gate)
    else:
        return discrete_mask(idx_array, gate) + ((L * gate - torch.floor(L * gate)) / L)

def l2_reg_ortho_32bit(mdl):
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
    
        self.index_array = torch.FloatTensor([[[ i for i in range(output_dim)]]]).to(device)

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

    def get_sparsity(self, no_reduction=False):
        if self.padding_idx != -1:
            floor_gates = torch.cat((self.gates.weight[:self.padding_idx], self.gates.weight[self.padding_idx+1:]))
        else:
            floor_gates = self.gates.weight
        if not no_reduction:
            return (1.0-floor_gates).mean()
        else:
            return (1.0-floor_gates)

    def get_sparsity_loss(self):
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight #+ 100 * l2_reg_ortho_32bit(self)

    def report(self):
        print(self.get_sparsity())
    
    def forward(self, input):
        data = self.embedding(input)
        gates = self.gates(input) * self.embedding.weight.size()[1]
        mask = get_mask(self.index_array, gates, grad_shape_func=self.grad_shaping)

        return data * mask

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
        self._init_func(self.gates.weight.data, a=0.001, b=1.0)
        self._init_func(self.weight.data)
        self._init_func(self.bias.data)

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def get_sparsity(self, no_reduction=False):
        floor_gates = self.gates.weight
        if not no_reduction:
            return (1.0-floor_gates).mean()
        else:
            return (1.0-floor_gates)

    def get_sparsity_loss(self):
        if self.sparsity is None:
            return 0.0
        else:
            return torch.norm(self.sparsity - self.get_sparsity(True), 2).mean() * self.reg_weight #+ 100 * l2_reg_ortho_32bit(self)

    def report(self):
        print(self.get_sparsity())
    
    def forward(self, input):

        all_ = torch.arange(self.vocab_size).to(input.device)
        gates = self.gates(all_) * self.weight.size()[0]
        mask = get_mask(self.index_array, gates, grad_shape_func=self.grad_shaping)
        mask = torch.transpose(mask, 0, 1)
        masked_weight = self.weight * mask
        return torch.matmul(input, masked_weight) + self.bias
