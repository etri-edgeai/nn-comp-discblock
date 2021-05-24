import copy
import pickle
import math
import collections

import numpy as np
import torch

from discblock.lowrank import compute_block_svd, compute_svd
from discblock.utils import score_to_block, find_min_rank_clustering, find_min_rank_scoring, make_blocks_from_gates, adjust_rank, count_parameters, walk, parse_score, make_clusters

def _block_loader(score_input, nblocks, target_sizes, embedding_names, dims, use_clusters=True, include_decoder=False, padding_idx=-1, alpha=0):
    assignments = []
    block_sizes = []
    scores = []
    if type(score_input) == str:
        if "p7" in score_input or "pth" in score_input or "mdl" in score_input or "ckpt" in score_input: # Torch model computed by differentiable scoring.
            with open(score_input, 'rb') as f:
                model = torch.load(f, map_location="cpu")
                for i, (target_size, embedding_name, dim) in enumerate(zip(target_sizes, embedding_names, dims)):
                    if type(model) == dict or type(model) == collections.OrderedDict:
                        if "state_dict" in model:
                            model = model["state_dict"]
                        elif "model" in model:
                            model = model["model"] # SST-1
                            model = model.state_dict()
                        elif "model_state" in model: # joey
                            model = model["model_state"]
                        embedding = model[embedding_name+".gates.weight"]
                        assignment, block_size, score  = make_blocks_from_gates(
                            embedding,
                            target_size,
                            nblocks,
                            dim,
                            use_clusters,
                            padding_idx=padding_idx,
                            alpha=alpha)
                    else:
                        embedding = walk(model, embedding_name.split("."))
                        assignment, block_size, score  = make_blocks_from_gates(
                            embedding.gates,
                            target_size,
                            nblocks,
                            dim,
                            use_clusters,
                            padding_idx=padding_idx,
                            alpha=alpha)
                    assignments.append(assignment)
                    block_sizes.append(block_size)
                    scores.append(score)

        else: # Pickle-saved file
            with open(score_input, "rb") as f:
                score_ = pickle.load(f)
                scores = parse_score(score_, len(target_sizes)==2)

            scores_ = []
            for i, (target_size, embedding_name, dim, score) in enumerate(zip(target_sizes, embedding_names, dims, scores)):
                if use_clusters:
                    assignment, block_size, score = make_clusters(score, target_size, nblocks, dim, padding_idx=padding_idx, alpha=alpha)
                else:
                    if padding_idx != -1:
                        score = score[:padding_idx] + score[padding_idx+1:] 
                    min_rank = find_min_rank_scoring(score, target_size, nblocks, dim, alpha=alpha)
                    assignment, block_size = score_to_block(score, min_rank, nblocks, alpha=alpha)
                scores_.append(score)
                assignments.append(assignment)
                block_sizes.append(block_size)
            scores = scores_

    elif isinstance(score_input, torch.nn.Module):
        for i, (target_size, embedding_name, dim) in enumerate(zip(target_sizes, embedding_names, dims)):
            embedding = walk(score_input, embedding_name.split("."))
            assignment, block_size, score  = make_blocks_from_gates(
                embedding.gates,
                target_size,
                nblocks,
                dim,
                use_clusters,
                padding_idx=padding_idx,
                alpha=alpha)
            assignments.append(assignment)
            block_sizes.append(block_size)
            scores.append(score)

    for block_size, dim in zip(block_sizes, dims):
        adjust_rank(block_size, dim)
    return assignments, block_sizes, scores

class EmbeddingMagic(object):

    def __init__(self,
                 mode,
                 embeddings,
                 target_ratio,
                 embedding_type=None,
                 options=None,
                 use_embedding_for_decoder=False,
                 device="cpu"
                 ):
        self.mode = mode
        if type(embeddings) == str:
            embeddings = (embeddings,)
        self.embeddings = embeddings
        self.target_ratio = target_ratio
        self.embedding_type = embedding_type
        self.options = options
        self.use_embedding_for_decoder = use_embedding_for_decoder
        self.device = device

    def convert_impl(self, model, weight_dict=None, setup_weights=False):
        """
        model has the structure of the original model, which menas it is a basemodel.

        """
        try:
            embeddings_ = [walk(model, embedding.split(".")) for embedding in self.embeddings]
            devices_ = [embedding.weight.device for embedding in embeddings_]
            ntokens = []
            dims = []
            for i, module in enumerate(embeddings_):
                ntokens.append(module.weight.size()[0])
                dims.append(module.weight.size()[1])
        except Exception as e:
            if self.embedding_type == "smallfry":
                pass
            else:
                print(e)
                raise ValueError()

        padding_idx = self.options["padding_idx"] if "padding_idx" in self.options else -1
        print("PADDING IDX:", padding_idx)
        if "block" in self.embedding_type:
            nblocks = self.options["nblocks"]
            score_input = self.options["score"]
            use_clusters = self.options["use_clusters"]
            refinement = self.options["refinement"]
            alpha = self.options["alpha"] if "alpha" in self.options else 0
            mem_efficient = self.options["mem_efficient"] if "mem_efficient" in self.options else False

            if weight_dict is not None and self.mode != "train": # Not first time run
                assert False # is it used?
                block_assignment = weight_dict[self.embeddings[0]+".block_assignment"]
                local_assignment = weight_dict[self.embeddings[0]+".local_assignment"]
                assignments = []
                assignments.append([
                    (idx, block_idx, local_idx) for idx, (block_idx, local_idx) in enumerate(zip(block_assignment, local_assignment))
                ])
                if len(self.embeddings) == 2:
                    assignments.append(weight_dict[self.embeddings[1]+".idx2bidx"])

                block_sizes = [[],[]]
                for bidx in  range(nblocks):
                    en_block = weight_dict[self.embeddings[0]+".block.%d" % bidx]
                    de_block = weight_dict[self.embeddings[1]+".block.%d" % bidx]
                    block_sizes[0].append(en_block.shape)
                    block_sizes[1].append(de_block.shape)
            else:
                target_sizes = []
                for dim, ntoken in zip(dims, ntokens):
                    target_size = int((1.0/self.target_ratio) * ntoken * dim) - 2 * ntoken # for assignments
                    target_sizes.append(target_size)

                assignments, block_sizes, scores = _block_loader(score_input,
                                                                 nblocks,
                                                                 target_sizes,
                                                                 self.embeddings,
                                                                 dims,
                                                                 use_clusters=use_clusters,
                                                                 include_decoder=len(target_sizes)==2,
                                                                 padding_idx=padding_idx,
                                                                 alpha=alpha)

                print(block_sizes)
                if self.embedding_type == "block":
                    # Finalize the block sizes with refined SVD
                    block_sizes_ = []
                    block_svds = []
                    assignments_ = []
                    for module, assignment, block_size, score, target_size in zip(
                        embeddings_, assignments, block_sizes, scores, target_sizes):
                        if padding_idx != -1:
                            oweight = torch.cat((module.weight.data[:padding_idx], module.weight.data[padding_idx+1:]))
                        else:
                            oweight = module.weight.data
                        assignment, block_svd = compute_block_svd(
                        oweight,
                        assignment,
                        block_size,
                        target_size=target_size,
                        score=score,
                        refinement=refinement,
                        tmax=100,
                        m_min=100,
                        mem_efficient=mem_efficient)
                        block_size = [u.shape for u, vt in block_svd]
                        block_sizes_.append(block_size)
                        assignments_.append(assignment)
                        block_svds.append(block_svd)
                    block_sizes = block_sizes_
                    assignments = assignments_

            print(block_sizes)
            from discblock.layers.mblock import BlockWiseEmbedding, BlockWiseEmbeddingClassifier # More efficient for low-rank approx.
            for i, (assignment, block_size, dim, embedding, score) in enumerate(zip(assignments, block_sizes, dims, self.embeddings, scores)):
                if i == 0 or (i == 1 and self.use_embedding_for_decoder):
                    module = BlockWiseEmbedding(assignment, block_size, dim, padding_idx=padding_idx)
                else:
                    module = BlockWiseEmbeddingClassifier(assignment, block_size, dim)
                embedding = embedding.split(".")
                holder = walk(model, embedding[:-1])
                module.to(devices_[i])
                setattr(holder, embedding[-1], module)
                if setup_weights:
                    omodule = embeddings_[i] # original embedding
                    blocks = []
                    for num, rank in block_size:
                        block_arr = torch.zeros((num, dim), dtype=torch.float)
                        blocks.append(block_arr)
                    for idx, block_idx, local_idx in assignment:
                        if padding_idx != -1 and (i == 0 or (i == 1 and self.use_embedding_for_decoder)):
                            if padding_idx <= idx:
                                idx = idx + 1
                        blocks[block_idx][local_idx] = omodule.weight.data[idx]
                    for bidx in range(len(block_size)):
                        module.blocks[bidx].weight.data = blocks[bidx]
                    if i == 1 and not self.use_embedding_for_decoder:
                        module.bias.data = embeddings_[i].bias.data
                    if padding_idx != -1:
                        module.padding_vec.data = omodule.weight.data[padding_idx]

                if "_" not in self.embedding_type:
                    from discblock.layers.sblock import SVDEmbedding
                    for bidx, (num, rank) in enumerate(block_size):
                        bmodule = SVDEmbedding(num, rank, dim)
                        if rank == dim:
                            continue
                        if setup_weights:
                            u, v = block_svds[i][bidx]
                            assert u.shape == bmodule.embedding.weight.size()
                            assert v.shape == bmodule.transformer.weight.size()
                            bmodule.embedding.weight.data = u
                            bmodule.transformer.weight.data = v
                        module.blocks[bidx] = bmodule
                else:
                    class_type_str = self.embedding_type.split("_")[1]
                    if "w2k" in class_type_str : # word2ket
                        from word2ket import EmbeddingKet, EmbeddingKetXS
                        embedding_class = EmbeddingKet if "w2k" == class_type_str else EmbeddingKetXS
                        min_, max_ = None, None
                        bscores = [ 0 for _ in range(len(block_size)) ]
                        for bidx, (num, _) in enumerate(block_size):
                            sum_ = 0
                            for widx, bidx_, _ in assignment:
                                if bidx == bidx_:
                                    sum_ += score[widx]
                            rank = sum_ / num
                            bscores[bidx] = rank
                            if min_ is None:
                                min_ = rank
                            if max_ is None:
                                max_ = rank
                            min_ = min(min_, rank)
                            max_ = max(max_, rank)
                        for bidx, (num, _) in enumerate(block_size):
                            order = self.options["word2ket_options"]["order"]
                            rank_ = self.options["word2ket_options"]["rank"]
                            if len(block_sizes) == 1:
                                rank = rank_
                            else:
                                rank = bscores[bidx]
                                print(num, rank, rank_ * rank / max_)
                                rank = int(rank_ * (rank / max_))
                                rank = max(rank, 1)
                                print(rank)
                            bmodule = embedding_class(ntoken, dim, order, rank)
                            module.blocks[bidx] = bmodule

                    elif class_type_str == "sf": # smallfry
                        from smallfry.quant_embedding import QuantEmbedding
                        embedding_class = QuantEmbedding
                        min_, max_ = None, None
                        for bidx, (num, rank) in enumerate(block_size):
                            if min_ is None:
                                min_ = rank
                            if max_ is None:
                                max_ = rank
                            min_ = min(min_, rank)
                            max_ = max(max_, rank)
                        for bidx, (num, rank) in enumerate(block_size):
                            if len(block_size) == 1:
                                bit_ = self.options["smallfry_options"]["nbit"]
                            else:
                                bit_ = (rank / max_) * self.options["smallfry_options"]["nbit"]
                                bit_ = bit_ * self.options["smallfry_options"]["alpha"]
                                bit_ = max(bit_, 1.0)
                                bit_ = pow(2, int(math.log(bit_, 2)))
                                bit_ = int(max(bit_, 1.0))
                                bit_ = min(bit_, self.options["smallfry_options"]["nbit"])
                            print(bidx, bit_)
                            q = QuantEmbedding(num_embeddings=num,    # vocabulary size
                                               embedding_dim=dim,       # embedding dimensionality
                                               _weight=module.blocks[bidx].weight,
                                               nbit=bit_)
                            module.blocks[bidx] = q

        elif "diff_embedding" in self.embedding_type:
            if "svd" in self.embedding_type:
                from discblock.layers.diff_embedding_svd import DifferentiableEmbedding, DifferentiableEmbeddingClassifier
            elif "ex" in self.embedding_type:
                from discblock.layers.diff_embedding_ex import DifferentiableEmbedding, DifferentiableEmbeddingClassifier
            elif "continuous" in self.embedding_type:
                from discblock.layers.diff_embedding_continuous import DifferentiableEmbedding, DifferentiableEmbeddingClassifier
            elif "non" in self.embedding_type:
                from discblock.layers.diff_embedding_non import DifferentiableEmbedding, DifferentiableEmbeddingClassifier
            else:
                from discblock.layers.diff_embedding import DifferentiableEmbedding, DifferentiableEmbeddingClassifier
            sparsity = self.options["sparsity"]
            reg_weight = self.options["reg_weight"]
            use_gumbel = "gumbel" in self.options and self.options["gumbel"]
            tau = self.options["tau"] if "tau" in self.options else 1
            core_dim = self.options["core_dim"] if "core_dim" in self.options else 1

            for key, val in self.options.items():
                print(key, "->", val)

            for i, (dim, embedding, ntoken) in enumerate(zip(dims, self.embeddings, ntokens)):
                if i == 0 or (i == 1 and self.use_embedding_for_decoder):
                    if "continuous" in self.embedding_type:
                        module = DifferentiableEmbedding(ntoken, dim, sparsity=sparsity, reg_weight=reg_weight, device=devices_[i], padding_idx=padding_idx, use_gumbel=use_gumbel, tau=tau, core_dim=core_dim)
                    else:
                        module = DifferentiableEmbedding(ntoken, dim, sparsity=sparsity, reg_weight=reg_weight, device=devices_[i], padding_idx=padding_idx)
                else:
                    if "continuous" in self.embedding_type:
                        module = DifferentiableEmbeddingClassifier(ntoken, dim, sparsity=sparsity, reg_weight=reg_weight, device=devices_[i], use_gumbel=use_gumbel, tau=tau, core_dim=core_dim)
                    else:
                        module = DifferentiableEmbeddingClassifier(ntoken, dim, sparsity=sparsity, reg_weight=reg_weight, device=devices_[i])
                module.to(devices_[i])
                embedding = embedding.split(".")
                holder = walk(model, embedding[:-1])
                setattr(holder, embedding[-1], module)

                module.init_weights()
                if setup_weights:
                    weight = embeddings_[i].weight
                    if type(module) == DifferentiableEmbedding:
                        if "svd" in self.embedding_type:
                            module.embedding.data = weight
                        else:
                            module.embedding.weight.data = weight
                    else:
                        if i == 1 and self.use_embedding_for_decoder:
                            module.weight.data = weight
                        else:
                            module.weight.data = torch.transpose(weight, 0, 1)
                    if "precomputed_gates" in self.options:
                        score_file = self.options["precomputed_gates"]
                        with open(score_file, "rb") as f:
                            score = pickle.load(f)
                        if len(score) == 2:
                            score = score[i]
                        
                        score_vec = [score[i_] for i_ in range(len(score))]
                        max_ = max(score_vec)
                        score_vec = [ val / max_ for val in score_vec ]
                        if padding_idx != -1 and (i == 0 or (i == 1 and self.use_embedding_for_decoder)):
                            score_vec[padding_idx] = 1.0 # max value for mask

                        t = torch.tensor(score_vec).to(devices_[i])
                        if "svd" in self.embedding_type:
                            if module.gates.size() != t.size():
                                t = t.view(module.gates.size())
                            module.gates.data = t
                        else:
                            if module.gates.weight.size() != t.size():
                                t = t.view(module.gates.weight.size())
                            module.gates.weight.data = t
                    if i == 1 and hasattr(embeddings_[i], "bias"):
                        module.bias.data = embeddings_[i].bias.data

            if self.options["gate_training_only"]:
                for name, p in model.named_parameters():
                    p.requires_grad = "gates" in name

        elif self.embedding_type == "svd":
            svd_rank = self.options["rank"]
            from discblock.layers.sblock import SVDEmbedding, SVDEmbeddingClassifier
            for i, (dim, embedding, ntoken) in enumerate(zip(dims, self.embeddings, ntokens)):
                if i == 0 or (i == 1 and self.use_embedding_for_decoder):
                    module = SVDEmbedding(ntoken, svd_rank, dim)
                else:
                    module = SVDEmbeddingClassifier(ntoken, svd_rank, dim)
                module.to(devices_[i])
                embedding = embedding.split(".")
                holder = walk(model, embedding[:-1])
                setattr(holder, embedding[-1], module)

                if setup_weights:
                    weight = embeddings_[i].weight
                    u, v = compute_svd(weight, svd_rank)
                    if i == 1 and not self.use_embedding_for_decoder:
                        v = v.transpose(0, 1)
                    assert u.shape == module.embedding.weight.size()
                    assert v.shape == module.transformer.weight.size()
                    module.embedding.weight.data = u
                    module.transformer.weight.data = v
                    if i == 1 and hasattr(embeddings_[i], "bias"):
                        module.bias.data = embeddings_[i].bias.data

        elif "word2ket" in self.embedding_type:
            from word2ket import EmbeddingKet, EmbeddingKetXS
            embedding_class = EmbeddingKet if "word2ket" == self.embedding_type else EmbeddingKetXS
            order = self.options["order"]
            rank = self.options["rank"]
            for i, (dim, embedding, ntoken) in enumerate(zip(dims, self.embeddings, ntokens)):
                module = embedding_class(ntoken, dim, order, rank)
                embedding = embedding.split(".")
                holder = walk(model, embedding[:-1])
                setattr(holder, embedding[-1], module)
            if setup_weights:
                pass

        elif "smallfry" in self.embedding_type:
            from smallfry.quant_embedding import quantize_embed
            nbit = self.options["nbit"]
            quantize_embed(model, nbit)
            if setup_weights:
                pass
        else:
            raise NotImplementedError()
        
    def profile(self, model):
        embeddings_ = [walk(model, embedding.split(".")) for embedding in self.embeddings]
        nparams = sum([
            count_parameters(embedding, substr="")
            for embedding in embeddings_
        ])
        print("word embeddings:", nparams)
        print("total:", count_parameters(model, substr=""))
   
    def convert(self, model, weight_dict=None, setup_weights=True):
        print("---- Before replacement ----")
        self.profile(model)
        self.convert_impl(model, weight_dict, setup_weights)
        model.to(self.device)
        print("---- After replacement ----")
        self.profile(model)

    @torch.no_grad()
    def evaluate_gates(self, dmodel, block_options, load_func, eval_func):
        assert "diff_embedding" in self.embedding_type 
        block_options["score"] = dmodel
        submagic = EmbeddingMagic(
            self.mode,
            self.embeddings,
            self.target_ratio,
            embedding_type="block",
            options=block_options,
            use_embedding_for_decoder=self.use_embedding_for_decoder)
        model = load_func()
        submagic.convert(model, setup_weights=True)
        model.to(self.device)
        return eval_func(model)
