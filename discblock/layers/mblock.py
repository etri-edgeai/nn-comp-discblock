""" Block-wise Embedding

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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockWiseEmbedding(nn.Module):
    """ Block-wise Embedding Layer """

    def __init__(
        self,
        assignment,
        block_sizes,
        output_dim,
        embedding_class=nn.Embedding,
        embedding_initializer=nn.init.normal_,
        padding_idx=-1,
        needs_restoration=False,
        **embedding_kwargs):
        super(BlockWiseEmbedding, self).__init__()

        block_assign_ = torch.zeros(len(assignment), requires_grad=False)
        local_idx_ = torch.zeros(len(assignment), requires_grad=False)
        for idx, block_idx, local_idx in assignment:
            block_assign_[idx] = block_idx
            local_idx_[idx] = local_idx
        self.register_buffer("block_assignment", block_assign_)
        self.register_buffer("local_assignment", local_idx_)

        self.blocks = nn.ModuleList([
            embedding_class(int(num), int(output_dim), **embedding_kwargs)
            for num, size in block_sizes
        ])

        self.embedding_initializer = embedding_initializer
        if padding_idx != -1:
            self.padding_vec = nn.Parameter(torch.FloatTensor(output_dim,), requires_grad=True)
        else:
            self.padding_vec = None
        self.padding_idx = padding_idx
        self.needs_restoration=needs_restoration

        self.temp_embedding = None # for knowledge embedding

    def init_weights(self):
        """ Initialize Weights """

        for b in self.blocks:
            if hasattr(b, "init_weights"):
                b.init_weights()
            elif self.embedding_initializer is not None:
                self.embedding_initializer(b.weight.data)

    def _restore(self):
        """ Restore function imple. """

        blocks_ = []
        for b in self.blocks:
            if hasattr(b, "transformer"):
                B = b.transformer(b.embedding.weight)
            else:
                B = b.weight
            blocks_.append(B)

        vecs = []
        for idx, (bidx, lidx) in enumerate(zip(self.block_assignment, self.local_assignment)):
            vecs.append(blocks_[int(bidx)][int(lidx)])
        self.temp_embedding = torch.stack(vecs)

    def restore(self, no_grad=False):
        """ Restore function """

        if no_grad:
            with torch.no_grad():
                self._restore()
        else:
            self._restore()

    def clear(self):
        """ Clear """
        self.temp_embedding = None

    def forward(self, src):
        """ Override forward function """

        # TODO: 100 should be changed later
        if self.temp_embedding is not None and len(src) > 100:
            return torch.nn.functional.embedding(src, self.temp_embedding)

        nobatch = False
        if len(src.shape) == 1:
            if self.needs_restoration:
                src_ = src
                src = torch.unique(src)
                inv = {}
                for i, val in enumerate(src):
                    inv[int(val)] = i
                rsrc = []
                for val in src_:
                    rsrc.append(inv[int(val)])
                rsrc = torch.LongTensor(rsrc).cuda()
            src = src.unsqueeze(0)
            nobatch = True

        bags = []
        collectors = [
            [] for _ in range(len(self.blocks))
        ]
        index = []
        for bidx, data in enumerate(src):
            sub_index = []
            for widx, idx in enumerate(data):
                idx = int(idx)
                if idx == self.padding_idx and self.padding_vec is not None:
                    sub_index.append((None, None))
                else:
                    if self.padding_vec is not None:
                        if idx > self.padding_idx:
                            idx = idx - 1

                    block_idx = int(self.block_assignment[idx])
                    local_idx = int(self.local_assignment[idx])
                    collectors[block_idx].append(local_idx)
                    sub_index.append((block_idx, len(collectors[block_idx])-1))
            index.append(sub_index)

        retrieved = []
        for block_idx, local_indices in enumerate(collectors):
            if len(local_indices) == 0:
                retrieved.append(None) # holder
                continue
            indices = torch.LongTensor(local_indices).to(src.device)
            vecs = self.blocks[block_idx](indices)
            retrieved.append(vecs)

        for bidx, sub_index in enumerate(index):
            vecs = []
            for widx, (block_idx, tidx) in enumerate(sub_index):
                if block_idx is None:
                    vec = self.padding_vec
                else:
                    vec = retrieved[block_idx][tidx]
                vecs.append(vec)
            bags.append(torch.stack(vecs))

        if nobatch:
            if self.needs_restoration:
                return torch.nn.functional.embedding(rsrc, bags[0])
            else:
                assert len(bags) == 1
                return bags[0]
        else:
            return torch.stack(bags)

class BlockWiseEmbeddingClassifier(nn.Module):
    """ Classifier Embedding """

    def __init__(
        self,
        assignment,
        block_sizes,
        input_dim,
        embedding_class=nn.Embedding,
        embedding_initializer=nn.init.normal_,
        bias_initializer=nn.init.uniform_,
        **embedding_kwargs):
        super(BlockWiseEmbeddingClassifier, self).__init__()

        if type(assignment) == list:
            idx2bidx_ = torch.zeros(len(assignment), requires_grad=False).to(torch.long)
            inv_mapping = {}
            for idx, block_idx, local_idx in assignment:
                inv_mapping[(block_idx, local_idx)] = idx

            block_serialized_idx = -1
            for bidx in range(len(block_sizes)):
                num, size = block_sizes[bidx]
                for lidx in range(num):
                    idx = inv_mapping[(bidx, lidx)]
                    block_serialized_idx += 1
                    idx2bidx_[idx] = block_serialized_idx
        else:
            idx2bidx_ = assignment
        self.register_buffer("idx2bidx", idx2bidx_)

        self.block_sizes = block_sizes
        self.blocks = nn.ModuleList([
            embedding_class(int(num), int(input_dim), **embedding_kwargs)
            for num, size in block_sizes
        ])
        self.bias = nn.Parameter(torch.FloatTensor(len(idx2bidx_),))

        self.embedding_initializer = embedding_initializer
        self.bias_initializer = bias_initializer

    def init_weights(self):
        """ Initialization """

        # init weights
        for b in self.blocks:
            if hasattr(b, "init_weights"):
                b.init_weights()
            elif self.embedding_initializer is not None:
                self.embedding_initializer(b.weight.data)
        self.bias_initializer(self.bias.data)

    def forward(self, src):
        """ Overriden Foward """

        ret = None
        # get embedding matrix
        for bidx, block in enumerate(self.blocks):
            all_ids = torch.arange(self.block_sizes[bidx][0]).to(src.device)
            B = block(all_ids)
            Bt = torch.transpose(B, 0, 1)
            temp = torch.matmul(src, Bt)

            if ret is None:
                ret = temp
            else:
                ret = torch.cat((ret, temp), 2)
        ret = ret[:, :, self.idx2bidx]
        return ret + self.bias
