#!/usr/bin/env python3

import yaml
import torch
import random
import argparse
import json
import numpy as np
import datetime

from pathlib import Path

from src.marcos import *
from src.mono_interface import MonoASRInterface
from src.utils import get_usable_cpu_cnt

# Make cudnn deterministic to reproduce result
torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='CommonVoice E2E ASR training/testing scripts')

# shared opts
parser.add_argument('--config', type=str,
                    help='Path to experiment config.', required=True)
parser.add_argument('--eval_suffix', type=str, default=None,
                    help='Evaluation suffix')
parser.add_argument('--runs', type=int, default=0)
parser.add_argument('--accent', choices=AVAIL_ACCENTS, required=True)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--seed', type=int, default=531, 
                    help='Random seed for reproducable results.')
parser.add_argument('--no_cuda',action='store_true')
parser.add_argument('--no_memmap',action='store_true')
parser.add_argument('--algo', choices=['reptile','fomaml', 'multi', 'fomaml_fast','no'], required=True)
parser.add_argument('--adv', action='store_true')
parser.add_argument('--model_name', choices=['blstm','las'], default='blstm')
parser.add_argument('--njobs', type=int, default=-1, 
                    help='Number of threads for decoding.')
parser.add_argument('--freeze_layer', type=str, default=None, choices=['VGG','VGG_BLSTM'])

parser.add_argument('--save_verbose', action='store_true')

# pretrain
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_suffix', type=str, default=None,
                    help='Pretrain model suffix')
parser.add_argument('--pretrain_setting', type=str, default=None)
parser.add_argument('--pretrain_runs', type=int, default=0)
parser.add_argument('--pretrain_step', type=int, default=0)
parser.add_argument('--pretrain_tgt_accent', choices=AVAIL_ACCENTS, default='107')
parser.add_argument('--pretrain_model_path',type=str, default=None, 
                    help='directly set Pretrain model path')

# training opts 
parser.add_argument('--resume',action='store_true')
parser.add_argument('--no_bucket',action='store_true')

# testing opts
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--test_model',type=str, default='model.cer.best', 
                    help='Evaluate on this model')
# parser.add_argument('--resume_decode_step', default=0, type=int)
# paser.add_argument('--decode_resume')
parser.add_argument('--decode_mode', choices=['greedy', 'beam', 'lm_beam'],
                    default='greedy')
parser.add_argument('--decode_suffix', default=None, type=str)  # will remove later
parser.add_argument('--lm_model_path', default=None, type=str)
# parser.add_argument('--nbest', default=5, type=int)

paras = parser.parse_args()
cur_time_suffix = "{:%B%d-%H%M%S}".format(datetime.datetime.now())
paras.eval_suffix = paras.eval_suffix if paras.eval_suffix else cur_time_suffix
paras.decode_suffix = f"{paras.decode_mode}_decode_{paras.decode_suffix}" if paras.decode_suffix else f"{paras.decode_mode}_decode" 

setattr(paras,'cuda', not paras.no_cuda)
setattr(paras,'is_bucket', not paras.no_bucket)
setattr(paras,'is_memmap', not paras.no_memmap)
if paras.adv:
    assert paras.algo != 'no'
    paras.algo += '-adv'

paras.njobs = paras.njobs if paras.njobs > 0 else get_usable_cpu_cnt()
config = yaml.safe_load(open(paras.config,'r'))

# Seed init.
random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)

with open(Path('data','accent-code.json'),'r') as fin:
    id2accent = json.load(fin)

if paras.test:
    from src.tester import Tester
    solver = Tester(config, paras, id2accent)
else:
    if paras.model_name == 'blstm':
        from src.blstm_trainer import get_trainer
    elif paras.model_name == 'las':
        from src.las_trainer import get_trainer
    else:
        raise NotImplementedError
    solver = get_trainer(MonoASRInterface, config, paras, id2accent)

solver.load_data()
solver.set_model()
solver.exec()
