from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDEmbedding(nn.Module):

    def __init__(
        self,
        num,
        rank,
        output_dim,
        embedding_initializer=nn.init.normal_,
        transformer_initializer=nn.init.xavier_uniform_):
        super(SVDEmbedding, self).__init__()

        self.embedding = nn.Embedding(int(num), int(rank))
        self.transformer = nn.Linear(int(rank), output_dim, bias=False)

        self.embedding_initializer = embedding_initializer
        self.transformer_initializer = transformer_initializer

    def init_weights(self):
        # init weights
        self.embedding_initializer(self.embedding.weight.data)
        self.transformer_initializer(self.transformer.weight.data)
        
    def forward(self, src):
        return self.transformer(self.embedding(src))

class SVDEmbeddingClassifier(nn.Module):

    def __init__(
        self,
        num,
        rank,
        input_dim,
        embedding_initializer=nn.init.normal_,
        transformer_initializer=nn.init.xavier_uniform_,
        bias_initializer=nn.init.uniform_):
        super(SVDEmbeddingClassifier, self).__init__()

        self.embedding = nn.Linear(int(rank), int(num))
        self.transformer = nn.Linear(input_dim, int(rank), bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(int(num),))

        self.embedding_initializer = embedding_initializer
        self.transformer_initializer = transformer_initializer
        self.bias_initializer = bias_initializer

    def init_weights(self):
        # init weights
        self.embedding_initializer(self.embedding.weight.data)
        self.transformer_initializer(self.transformer.weight.data)
        self.bias_initializer(self.bias.data)
        
    def forward(self, src):
        return self.embedding(self.transformer(src)) + self.bias
