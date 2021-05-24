import copy

import numpy as np
import torch
from scipy.sparse.linalg import svds

def compute_svd(embedding, rank, q=None, mem_efficient=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
    embedding = embedding.cpu()
    if q is not None:
        q = q.cpu()
        if mem_efficient:
            q = torch.sqrt(q)
            for idx in range(embedding.size()[0]):
                embedding[idx] = q[idx] * embedding[idx]
        else:
            Q = torch.diag(torch.sqrt(q))
            embedding = torch.matmul(Q, embedding)
    u, s, v = torch.svd_lowrank(embedding, q=int(rank))

    if q is not None:
        if mem_efficient:
            iq = torch.reciprocal(q) # q was already changed to torch.sqrt(q).
            for idx in range(u.size()[0]):
                u[idx] = iq[idx] * u[idx]
            u_star = torch.matmul(u, torch.diag(s))
        else:
            inv_Q = torch.diag(torch.reciprocal(torch.sqrt(q)))
            u_star = torch.matmul(torch.matmul(inv_Q, u), torch.diag(s))
    else:
        u_star = torch.matmul(u, torch.diag(s))
    return (u_star, v)

def _argmin_reconstruct_error(w, block_svds):
    min_ = -1
    min_idx = -1
    for idx, (u, v) in enumerate(block_svds):
        if v is None:
            err = 0.0
        else:
            err = w - torch.matmul(torch.matmul(v, v.transpose(0,1)), w)
            err = torch.pow(torch.norm(err, p=2), 2)
        if min_ == -1 or min_ > err:
            min_ = err
            min_idx = idx
    return min_idx, min_

def _construct_embedding(block, len_, dtype):
    block_arr = torch.zeros((len(block), len_), dtype=dtype)
    keys = sorted(list(block.keys()))
    for idx, key in enumerate(keys):
        block_arr[idx] = block[key]
    return keys, block_arr

def _compute_single_block_svd(block, keys, rank, score=None, mem_efficient=False):
    if score is not None:
        q = torch.tensor([score[key] for key in keys]).to(block.device).float()
        u, v = compute_svd(block, rank, q, mem_efficient)
    else:
        u, v = compute_svd(block, rank, mem_efficient)
    return (u, v)

def size(block_svds):
    nparams = 0
    for u, vt in block_svds:
        if vt is not None:
            nparams += np.prod(u.shape) + np.prod(vt.shape)
        else:
            nparams += np.prod(u.shape)
    return nparams

def size_v2(block_sizes, emsize):
    nparams = 0
    for num, rank in block_sizes:
        if rank == emsize:
            nparams += num * rank
        else:
            nparams += num * rank + rank * emsize
    return nparams

def refine_by_moving(embedding, block_svds, blocks, block_assignment, block_sizes, score, target_size, moving_ratio, tmax, m_min, mem_efficient=False):
    len_ = embedding.shape[0]
    dim = embedding.shape[1]

    for t in range(tmax):
        print(t, tmax)
        marker = [False for _ in range(len(blocks))]
        candidates = []
        osize = size(block_svds)
        for i in range(len_):
            w = embedding[i]
            min_idx, min_err = _argmin_reconstruct_error(w, block_svds)
            if block_assignment[i] != min_idx:
                candidates.append((i, min_idx, min_err))
        if len(candidates) < m_min:
            break
        candidates = sorted(candidates, key=lambda x: x[2])[: int(len(candidates) * moving_ratio)]
        # move
        is_updated = False
        temp_change = 0
        for i, min_idx, min_err in candidates:
            if target_size is not None:
                # Check if moving violates the size constraint.
                orank = block_sizes[block_assignment[i]][1]
                nrank = block_sizes[min_idx][1]
                if (nrank - orank) + osize + temp_change > target_size:
                    break
                else:
                    temp_change += nrank - orank
            
            blocks[block_assignment[i]].pop(i)
            marker[block_assignment[i]] = True
            block_assignment[i] = min_idx
            blocks[min_idx][i] = embedding[i]
            marker[min_idx] = True
            is_updated = True

        if not is_updated:
            break
       
        block_svds_ = []
        for b in range(len(blocks)):
            block = blocks[b]
            if marker[b]:
                keys, block_ = _construct_embedding(block, dim, dtype=embedding.dtype)
                _, rank = block_sizes[b]
                if rank == dim:
                    block_svds_.append((block_, None))
                else:
                    block_svds_.append(_compute_single_block_svd(block_.to(embedding.device), keys, rank, score=score, mem_efficient=mem_efficient))
            else:
                block_svds_.append(block_svds[b])
        block_svds = block_svds_
    return block_svds

def refine_by_expanding(embedding, blocks, block_sizes, score, target_size, mem_efficient=False):
    dim = embedding.shape[1]

    # sort by num
    blocks_ = [ (bidx, num, rank) for bidx, (num, rank) in enumerate(block_sizes) ]
    blocks_ = sorted(blocks_, key=lambda x: x[1], reverse=True)

    block_sizes_ = copy.deepcopy(block_sizes)
    for bidx, num, rank in blocks_: # blocks are sorted by `num`
        if rank == dim:
            continue
        # compute the maximal rank.
        size = size_v2(block_sizes_, dim)
        excluded = size - (num * rank + rank * dim)
        permitted_size = target_size - excluded
        new_rank = int(permitted_size / (num + dim))

        print(bidx, size, target_size, permitted_size, excluded, new_rank, rank)
        if rank >= new_rank:
            continue
        else:
            assert new_rank > rank
            block_sizes_[bidx] = (num, min(num, new_rank))

    block_svds_ = []
    for b in range(len(blocks)):
        block = blocks[b]
        keys, block_ = _construct_embedding(block, dim, dtype=embedding.dtype)
        _, rank = block_sizes_[b]
        if rank == dim:
            block_svds_.append((block_, None))
        else:
            block_svds_.append(_compute_single_block_svd(block_.to(embedding.device), keys, rank, score=score, mem_efficient=mem_efficient))
    return block_svds_

def compute_block_svd(embedding, assignment, block_sizes, target_size=None, score=None, refinement=False, tmax=10, m_min=100, moving_ratio=0.1, mem_efficient=False):

    min_rank = None
    for _, rank in block_sizes:
        if min_rank is None or min_rank > rank:
            min_rank = rank # must be same as config["block_options"]["min_rank"]

    with torch.no_grad():
        len_ = embedding.shape[0]
        dim = embedding.shape[1]

        # make blocks
        block_assignment = {}
        blocks = {}
        for idx, block_idx, _ in assignment:
            block_assignment[idx] = block_idx
            if block_idx not in blocks:
                blocks[block_idx] = {}
            blocks[block_idx][idx] = embedding[idx]

        block_svds = [None for _ in range(len(blocks))]
        block_embeddings = {}
        for bidx in blocks:
            keys, block_ = _construct_embedding(blocks[bidx], dim, dtype=embedding.dtype)
            block_embeddings[bidx] = (keys, block_)
            _, rank = block_sizes[bidx]
            if rank == dim:
                block_svds[bidx] = (block_, None)
            else:
                block_svds[bidx] = _compute_single_block_svd(block_.to(embedding.device), keys, rank, score=score, mem_efficient=mem_efficient)

        if refinement:
            #block_svds = refine_by_moving(embedding, block_svds, blocks, block_assignment, block_sizes, score, target_size, moving_ratio, tmax, m_min)
            block_svds = refine_by_expanding(embedding, blocks, block_sizes, score, target_size, mem_efficient=mem_efficient)

        local_assignment = {}
        for bidx in blocks:
            keys = sorted(list(blocks[bidx].keys()))
            for local_idx, key in enumerate(keys):
                local_assignment[key] = local_idx

        new_assignment = []
        for idx, _, _ in assignment:
            new_assignment.append((idx, block_assignment[idx], local_assignment[idx]))
        return new_assignment, block_svds
