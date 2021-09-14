import importlib.util
spec = importlib.util.spec_from_file_location("compute_tfidf", "../../tools/compute_tfidf.py")
compute_tfidf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_tfidf)

import argparse

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Dataset

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.training import TrainManager
from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, latest_checkpoint_update, \
    ConfigurationError
from joeynmt.model import Model, _DataParallel
from joeynmt.prediction import validate_on_data, parse_test_args
from joeynmt.loss import XentLoss
from joeynmt.data import load_data, make_data_iter
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN

ap = argparse.ArgumentParser("Joey NMT")
ap.add_argument("config_path", type=str,
                help="path to YAML config file")
ap.add_argument('--mode', type=str, default="tfidf", help="mode")
args = ap.parse_args()
cfg = load_config(args.config_path)

# load the data
train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
    data_cfg=cfg["data"])

pad_index = trg_vocab.stoi[PAD_TOKEN]
print("PAD_INDEX:", pad_index)
train_iter = make_data_iter(train_data,
                                 batch_size=80,
                                 batch_type="sentence",
                                 train=True,
                                 shuffle=False)

docs = []
t_docs = []
for idx, batch in enumerate(iter(train_iter)):
    # create a Batch object from torchtext batch
    batch = Batch(batch, pad_index, use_cuda=False)
    data = vars(batch)

    batch_ = []
    for sent in data["src"]:
        line_ = []
        for word in sent:
            line_.append(int(word))
        batch_.append(line_)
    docs.append(batch_)

    batch_ = []
    for sent in data["trg"]:
        line_ = []
        for word in sent:
            line_.append(int(word))
        batch_.append(line_)
    t_docs.append(batch_)

compute_tfidf.compute(len(src_vocab), docs, "joey", args.mode, pad_token=pad_index, alpha=0.0, beta=0.1, t_ntokens=len(trg_vocab), t_docs=t_docs, base=32)
