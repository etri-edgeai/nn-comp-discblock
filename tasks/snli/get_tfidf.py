import importlib.util
spec = importlib.util.spec_from_file_location("compute_tfidf", "../../tools/compute_tfidf.py")
compute_tfidf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_tfidf)

import datasets
from utils import *

args = parse_args()
mode = "frequency"
#mode = "tfidf"

dataset_options = {
                        'batch_size': args.batch_size, 
                        'device': "cpu"
                }
print(datasets.__dict__.keys())
dataset = datasets.__dict__[args.dataset](dataset_options)

dataset.train_iter.init_epoch()

docs = []

# it starts with 1.
for batch_idx, batch in enumerate(dataset.train_iter):
    batch_ = []
    for sent in batch.premise:
        line_ = []
        for word in sent:
            line_.append(int(word))
        batch_.append(line_)
    for sent in batch.hypothesis:
        line_ = []
        for word in sent:
            line_.append(int(word))
        batch_.append(line_)
    docs.append(batch_)

compute_tfidf.compute(33931, docs, "snli", mode, pad_token=1, alpha=0.0, beta=0.1)
