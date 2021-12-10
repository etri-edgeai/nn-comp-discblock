# Block-wise Word Embedding Compression Revisited: Better Weighting and Structuring

### Overview
This repository contains the official implementation for the paper, "Block-wise Word Embedding Compression Revisited: Better Weighting and Structuring", Findings of EMNLP 2021.
It contains all the implementations for the paper, and we are working on 100% reproduction with yaml-based configuration.

### Requirements
* python3, torch, scipy, numpy for discblock. For each task, you need to install required packages.

### Getting Started
This repository provides various tasks for experiments.
Let us show how to apply our method with language modeling.

#### Step 1: Train a base model
```
$ cd experiments/lm/medium
$ python ../../../run_lm.py --config base.yaml
```

#### Step 2: Compute the TF-IDF based Word Importance Score

#### Step 3: Compute Differentiable Word importance Score

#### Step 4: Block-wise Embedding Compression and Fine-tuning

### Tasks

#### Language Modeling

#### Neural Machine Translation

#### SNLI

#### SST-5

#### SQuAD

#### Knowledge Embedding

### Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).
