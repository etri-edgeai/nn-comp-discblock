import torch

from datasets import *
from config import Config

import json
import argparse

import importlib.util
spec = importlib.util.spec_from_file_location("compute_tfidf", "../../tools/compute_tfidf.py")
compute_tfidf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_tfidf)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', type=str, default="tfidf", help="mode")
args = parser.parse_args()

class ModelConfig(object):
    '''
    模型配置参数
    '''
    # 全局配置参数
    opt = Config()

    # 数据参数
    output_folder = opt.output_folder
    data_name = "SST-1"
    SST_path  = opt.SST_path
    emb_file = opt.emb_file
    emb_format = opt.emb_format
    output_folder = opt.output_folder
    min_word_freq = opt.min_word_freq
    max_len = opt.max_len

    # 训练参数
    epochs = 120  # epoch数目，除非early stopping, 先开20个epoch不微调,再开多点epoch微调
    batch_size = 32 # batch_size
    workers = 4  # 多处理器加载数据
    lr = 1e-4  # 如果要微调时，学习率要小于1e-3,因为已经是很优化的了，不用这么大的学习率
    weight_decay = 1e-5 # 权重衰减率
    decay_epoch = 20 # 多少个epoch后执行学习率衰减
    improvement_epoch = 6 # 多少个epoch后执行early stopping
    is_Linux = True # 如果是Linux则设置为True,否则设置为else, 用于判断是否多处理器加载
    print_freq = 100  # 每隔print_freq个iteration打印状态
    checkpoint = None  # 模型断点所在位置, 无则None
    best_model = None # 最优模型所在位置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    model_name = 'TextCNN' # 模型名
    class_num = 5 if data_name == 'SST-1' else 2 # 分类类别
    kernel_num = 100 # kernel数量
    kernel_sizes = [3,4,5] # 不同尺寸的kernel
    dropout = 0.5 # dropout
    embed_dim = 128 # 未使用预训练词向量的默认值
    static = True # 是否使用预训练词向量, static=True, 表示使用预训练词向量
    non_static = True # 是否微调，non_static=True,表示微调
    multichannel = True # 是否多通道

opt = ModelConfig()
train_loader = torch.utils.data.DataLoader(
                SSTreebankDataset(opt.data_name, opt.output_folder, 'train'),
                batch_size=opt.batch_size, 
                shuffle=True,
                num_workers = opt.workers if opt.is_Linux else 0,
                pin_memory=True)

word_map_file = opt.output_folder +  opt.data_name + '_' + 'wordmap.json'
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

docs = []
for i, (sents, labels) in enumerate(train_loader):
    batch_ = []
    for sent in sents:
        line_ = []
        for word in sent:
            line_.append(int(word))
        batch_.append(line_)
    docs.append(batch_)

compute_tfidf.compute(
    len(word_map), docs, "sst-1", args.mode, pad_token=0, alpha=0.0, beta=0.1)
