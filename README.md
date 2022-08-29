# Block-wise Word Embedding Compression Revisited: Better Weighting and Structuring

This repository contains the official implementation for the paper, ["Block-wise Word Embedding Compression Revisited: Better Weighting and Structuring", Findings of EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.372).
It contains all the implementations for the paper, and we are working on 100% reproduction with yaml-based configuration for convenience.

### Abstract
Word embedding is essential for neural network models for various natural language processing tasks.
Since the word embedding usually has a considerable size, in order to deploy a neural network model having it on edge devices, it should be effectively compressed.
There was a study for proposing a block-wise low-rank approximation method for word embedding, called GroupReduce.
Even if their structure is effective, the properties behind the concept of the block-wise word embedding compression were not sufficiently explored.
Motivated by this, we improve GroupReduce in terms of word weighting and structuring.
For word weighting, we propose a simple yet effective method inspired by the term frequency-inverse document frequency method and 
a novel differentiable method.
Based on them, we construct a discriminative word embedding compression algorithm.
In the experiments, we demonstrate that the proposed algorithm more effectively finds word weights than competitors in most cases.
In addition, we show that the proposed algorithm can act like a framework through successful cooperation with quantization.

<div align="center">
  <img src="https://github.com/etri-edgeai/nn-comp-discblock/blob/main/asset/aligned_masking.PNG?raw=true" width="550px" />
</div>

<div align="center">
  <img src="https://github.com/etri-edgeai/nn-comp-discblock/blob/main/asset/conpensation.PNG?raw=true" width="850px" />
</div>

### Requirements
* For Language Modeling <br />
torch == 1.7.1 <br />
torchtext == 0.8.1 <br />
yaml <br />
scipy <br />
numpy <br />

### Getting Started
Let us show how to apply our method with language modeling.

#### Step 0: Download the model file
Download the model file from PyTorch's examples.
```
$ cd tasks/lm
$ bash download.bash
```

#### Step 1: Train a base model
```
$ cd experiments/lm/medium
$ python ../../../run_lm.py --config base.yaml
```

#### Step 2: Compute the TF-IDF based Word Importance Score
```
$ cd tasks/lm
$ python get_tfidf.py --data ptb --model tfidf
```
The example dataset is PennTree Bank and the TF-IDF based scoring scheme is used.

#### Step 3: Compute Differentiable Word importance Score
```
$ cd experiment/lm/diff
$ python ../../../run_lm.py --config config.yaml
```
The path to the base model and all hyper-parameters are already set in `config.yaml`.

#### Step 4: Block-wise Embedding Compression and Fine-tuning
```
$ cd experiment/lm/diff_score
$ python ../../../run_lm.py --config config.yaml
```
In the same way, all the hyper-parameters are already set in `config.yaml`.

### Compression Performance (about 20x compression ratio)

#### Before Retraining

| Datasets     |      Original PPL	      |	SVD						|	DisckBlock-F	|  DiscBlock-T |	DiscBlock-D	|
|--------------|:------------------------:|--------------:|--------------:|-------------:|-------------:|
| 	PTB	 	     |						80.8				  |	372.1					|	156.6					| 136.4 			 |		125.7			|
| WikiText2    |   					93.3				  |	1,246.9				|	172.4					| 150.6 			 |		139.3			|
| WikiText103  | 						61.0				  |	1,882.1				| 122.2					| 92.6 				 |		75.3			|

#### After Retraining

| Datasets     |      Original PPL	      |	SVD						|	DisckBlock-F	|  DiscBlock-T |	DiscBlock-D	|
|--------------|:------------------------:|--------------:|--------------:|-------------:|-------------:|
| 	PTB	 	     |						80.8				  |	96.0					|	92.9					| 92.0 				 |		88.7			|
| WikiText2    |   					93.3				  |	115.0					|	107.8					| 104.3 			 |		102.7			|
| WikiText103  | 						61.0				  |	83.2					|	82.5					| 72.4 				 |		67.6			|

### Tasks
We provide the implementation for various tasks introduced in the paper here.
Note that we have excluded codes originated from third-party repositories which do not have any license term.

##### Language Modeling
Our implementation for language modeling is based on https://github.com/pytorch/examples/tree/master/word_language_model.
For handling this task, use `run_lm.py` and see `tasks/lm`.

#### SNLI, SST, Knowledge Embedding, Neural Machine Translation, SQuAd
Due to the internal opensource policy of our company, we do not include implementations for the other tasks in this release.
Instead, we will include detailed explanation to a way of using DiscBlock for your custom task.

### Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).

### Citation

```BibTeX
@inproceedings{lee-etal-2021-block-wise,
    title = "Block-wise Word Embedding Compression Revisited: Better Weighting and Structuring",
    author = "Lee, Jong-Ryul  and
      Lee, Yong-Ju  and
      Moon, Yong-Hyuk",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "4379--4388"
}
```
