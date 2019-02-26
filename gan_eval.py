import os
import sys
from read_data import index_ent_rel, graph_size, read_data
from config import config, overwrite_config_with_args
from logger_init import logger_init
from data_utils import inplace_shuffle, heads_tails
from select_gpu import select_gpu
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from compl_ex import ComplEx
from simplE import SimplE
import torch
from config import config
from torch.autograd import Variable
from metrics import mrr_mr_hitk2
from data_utils import batch_by_size
import logging

logger_init()
overwrite_config_with_args()
torch.cuda.set_device(select_gpu())
task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]

models = {'TransE': TransE, 'TransD': TransD, 'DistMult': DistMult, 'ComplEx': ComplEx, 'SimplE': SimplE}

gen_config = config()[config().g_config]
dis_config = config()[config().d_config]
gan_config = config().kbgan_config
gen = models[config().g_config](n_ent, n_rel, gen_config)
dis = models[config().d_config](n_ent, n_rel, dis_config)

mdl_name = gan_config



if __name__ == '__main__':
  dis.load(os.path.join(config().task.dir, mdl_name))
  dis.eval_link(test_data, n_ent, heads, tails)