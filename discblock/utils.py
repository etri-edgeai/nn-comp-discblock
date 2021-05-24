import copy
import pickle

import torch
import numpy as np
from scipy.cluster.vq import kmeans2

DECODER_REQUIRED = ["lm"]

def adjust_rank(block_sizes, dim):
    for idx in range(len(block_sizes)):
        num, rank = block_sizes[idx]
        block_sizes[idx] = list(block_sizes[idx])
        if num == rank:
            block_sizes[idx][1] = dim

def compute_eps(min_, max_, alpha):
    if alpha == 0:
        return 0

    if alpha <= min_ / max_:
        print(alpha, min_ / max_)
    assert alpha >= min_ / max_
    return alpha * max_ - min_

def score_to_block(score, min_rank, nblock, assignment=None, alpha=0):
    score_ = sorted([(i, val) for i, val in enumerate(score)], key=lambda x: x[1], reverse=True)
    ntokens = len(score_)

    if assignment is None:
        assignment_ = []
        block_cnt = [ 0 for _ in range(nblock) ]
        for idx, (i, _) in enumerate(score_):
            bidx = idx // (ntokens // nblock)
            if bidx >= nblock:
                bidx -= 1
            local_idx = block_cnt[bidx]
            block_cnt[bidx] += 1
            assignment_.append((i, bidx, local_idx))
        assignment = assignment_
    else:
        assignment = copy.deepcopy(assignment)
   
    block_score = [ 0 for _ in range(nblock) ]
    block_cnt = [ 0 for _ in range(nblock) ]
    for idx, block_idx, _ in assignment:
        block_score[block_idx] += score[idx]
        block_cnt[block_idx] += 1

    min_score = -1
    max_score= -1
    for i in range(nblock):
        block_score[i] = float(block_score[i]) / block_cnt[i]
        if min_score == -1 or min_score > block_score[i]:
            min_score = block_score[i]
        if max_score == -1 or max_score < block_score[i]:
            max_score = block_score[i]

    block_sizes = []
    for i in range(nblock):
        if block_score[i] == min_score:
            score_ = block_score[i] + compute_eps(min_score, max_score, alpha)
        else:
            score_ = block_score[i]
        rank = min_rank * (score_ / (min_score + compute_eps(min_score, max_score, alpha)))
        block_sizes.append([block_cnt[i], min(block_cnt[i], int(rank))])

    return assignment, block_sizes

def find_min_rank_clustering(cluster_info, target_size, nblocks, dim, alpha=0):
    min_rank = 1
    block_cnt = [0 for _ in range(len(cluster_info[0]))]
    for idx, bidx in enumerate(cluster_info[1]):
        block_cnt[bidx] += 1
    block_size_ = [
        [val, cluster_info[0][idx]]
        for idx, val in enumerate(block_cnt)
    ]
    while True:

        block_size = copy.deepcopy(block_size_)
        min_ = None
        max_ = None
        for i in range(nblocks):
            if min_ is None or min_ > block_size[i][1]:
                assert block_size[i][1] >= 0
                min_ = block_size[i][1]
            if max_ is None or max_ < block_size[i][1]:
                assert block_size[i][1] >= 0
                max_ = block_size[i][1]

        for i in range(nblocks):
            if block_size[i][1] == min_:
               block_size[i][1] = min_ + compute_eps(min_, max_, alpha)
            block_size[i][1] = min(block_size[i][0], int((block_size[i][1] / (min_ + compute_eps(min_, max_, alpha))) * min_rank))
        nparams = sum([
            num * rank + rank * dim if num != rank else num * dim
        for num, rank in block_size])

        if target_size < nparams:
            break
        else:
            min_rank += 1
    return min_rank - 1

def find_min_rank_scoring(score, target_size, nblocks, dim, assignment=None, alpha=0):
    min_rank = 1
    while True:
        assignment, block_sizes = score_to_block(score, min_rank, nblocks, assignment=assignment, alpha=alpha)
        for i in range(nblocks):
            block_sizes[i][1] = min(block_sizes[i][0], block_sizes[i][1])

        nparams = sum([
            num * rank + rank * dim if num != rank else num * dim
        for num, rank in block_sizes])

        if target_size < nparams:
            break
        else:
            min_rank += 1
    return min_rank - 1

def make_clusters(score, target_size, nblocks, dim, padding_idx=-1, alpha=0):
    if padding_idx != -1:
        assert type(score) == list
        score = score[:padding_idx] + score[padding_idx+1:]

    if type(score[0]) == int:
        score = [float(v) for v in score]
    kmeans_ = kmeans2(score, nblocks, iter=1000, minit="points")
    min_rank = find_min_rank_clustering(kmeans_, target_size, nblocks, dim, alpha=alpha)

    assignment = []
    block_cnt = [0 for _ in range(len(kmeans_[0]))]
    for idx, bidx in enumerate(kmeans_[1]):
        local_idx = block_cnt[bidx]
        block_cnt[bidx] += 1
        assignment.append((idx, bidx, local_idx))
    block_sizes = [
        [val, kmeans_[0][idx]]
        for idx, val in enumerate(block_cnt)
    ]
    min_ = None
    max_ = None
    for i in range(nblocks):
        if min_ is None or min_ > block_sizes[i][1]:
            min_ = block_sizes[i][1]
        if max_ is None or max_ < block_sizes[i][1]:
            max_ = block_sizes[i][1]
    for i in range(nblocks):
        if block_sizes[i][1] == min_:
            block_sizes[i][1] = min_ + compute_eps(min_, max_, alpha)
        block_sizes[i][1] = min(block_sizes[i][0], int((block_sizes[i][1] / (min_ + compute_eps(min_, max_, alpha))) * (min_rank)))

    return assignment, block_sizes, score

def make_blocks_from_gates(gates, target_size, nblocks, dim, use_clusters, padding_idx=-1, alpha=0):
    if hasattr(gates, "weight"):
        index = gates.weight.detach().cpu().numpy()
    else:
        index = gates.detach().cpu().numpy()
    if len(index.shape) > 1 and index.shape[1] > 1:
        index = np.sum((index >= 0.5).astype(np.float32), axis=1)
    score = list(index.reshape(index.shape[0],))

    if use_clusters:
        assignment, block_sizes, score = make_clusters(score, target_size, nblocks, dim, padding_idx=padding_idx, alpha=alpha)
    else:
        if padding_idx != -1:
            score = score[:padding_idx] + score[padding_idx+1:]
        min_rank = find_min_rank_scoring(score, target_size, nblocks, dim, alpha=alpha)
        assignment, block_sizes = score_to_block(score, min_rank, nblocks, alpha=alpha)
    return assignment, block_sizes, score

def count_parameters(model, substr=""):
    dict_ = model.state_dict()
    sum_ = 0
    for key, val in dict_.items():
        #print(key, np.prod(val.shape), val.shape)
        sum_ += np.prod(val.shape)
    return sum_
    #return sum(p.numel() for name, p in model.named_parameters() if substr in name)

def walk(model, path):
    item = model
    for attr in path:
        item = getattr(item, attr)
    return item

def parse_score(score, include_decoder=False):
    if type(score) == tuple:
        scores = list(score)
    elif include_decoder:
        scores = [score, score]
    else:
        scores = [score]
    return scores

def compute_sparsity_loss(model):
    sparsity_loss = 0.0
    for c in model.modules():
        if type(c).__name__ == "DifferentiableEmbedding" or type(c).__name__ == "DifferentiableEmbeddingClassifier":
            sparsity_loss += c.get_sparsity_loss()
    return sparsity_loss
