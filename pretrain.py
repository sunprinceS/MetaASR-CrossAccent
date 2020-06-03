#!/usr/bin/env python3
import comet_ml
import yaml
import torch
import random
import argparse
import json

import numpy as np
import datetime
from pathlib import Path

from src.marcos import *
from src.utils import get_usable_cpu_cnt

# Make cudnn deterministic to reproduce result
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Accent-Adaptative ASR pretraining scripts')

parser.add_argument('--config', type=str, 
                    help='Path to experiment config.', required=True)
parser.add_argument('--pretrain_suffix', type=str, help='Pretrain model suffix', required=True)
parser.add_argument('--pretrain_accents', type=str, nargs='+', choices=AVAIL_ACCENTS)
parser.add_argument('--num_pretrain', type=int, required=True)
parser.add_argument('--tgt_accent', type=str, choices=AVAIL_ACCENTS)
parser.add_argument('--runs', default=0,type=int, help='Different runs means use different seed')
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--seed', default=531, type=int, help='Random seed for reproducable results.')
parser.add_argument('--no_cuda',action='store_true')
parser.add_argument('--no_memmap',action='store_true')
parser.add_argument('--no_bucket',action='store_true')

parser.add_argument('--meta_k', default=None, type=int)
parser.add_argument('--meta_batch_size', default=None, type=int)
parser.add_argument('--sample_strategy', default='normal', choices=['normal','meta-split','meta-split-dev'])
parser.add_argument('--max_step', default=0, type=int)
parser.add_argument('--resume',action='store_true')
parser.add_argument('--resume_step', default=-1, type=int)
parser.add_argument('--use_tensorboard',action='store_true')

parser.add_argument('--model_name', default='transformer', choices=['blstm', 'transformer'])
parser.add_argument('--algo', choices=['reptile','fomaml', 'multi', 'maml'], required=True)

parser.add_argument('--njobs', default=-1, type=int, help='Number of threads for decoding.')

# NOTE: Evaluation offline
paras = parser.parse_args()
cur_time_suffix = "{:%B%d-%H%M%S}".format(datetime.datetime.now())
paras.pretrain_suffix = paras.pretrain_suffix if paras.pretrain_suffix else cur_time_suffix

setattr(paras,'cuda', not paras.no_cuda)
setattr(paras,'is_bucket', not paras.no_bucket)
setattr(paras,'is_memmap', not paras.no_memmap)
meta_batch_size = paras.num_pretrain if paras.meta_batch_size is None else paras.meta_batch_size
assert meta_batch_size <= paras.num_pretrain, f"Meta batch size {meta_batch_size} > Number of pretraining accents {paras.num_pretrain}"
setattr(paras, 'meta_batch_size', meta_batch_size)
paras.njobs = paras.njobs if paras.njobs > 0 else get_usable_cpu_cnt()
config = yaml.safe_load(open(paras.config,'r'))

# Seed init.
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

with open(Path('data','accent-code.json'),'r') as fin:
    id2accent = json.load(fin)

if paras.algo == 'multi':
    from src.multi_interface import MultiASRInterface as ASRInterface
elif paras.algo == 'fomaml' or paras.algo == 'reptile':
    from src.fo_meta_interface import FOMetaASRInterface as ASRInterface
else:
    raise NotImplementedError

if paras.model_name == 'transformer':
    from src.transformer_torch_trainer import get_trainer
elif paras.model_name == 'blstm':
    raise NotImplementedError
else:
    raise NotImplementedError


solver = get_trainer(ASRInterface, config, paras, id2accent)
solver.load_data()
solver.set_model()
solver.exec()
