import pickle
import time

from shutil import rmtree
from pathlib import Path
import sentencepiece as spmlib

from tqdm import tqdm
from src.marcos import *
from src.io.dataset import get_loader
import src.monitor.logger as logger
from src.monitor.dashboard import Dashboard
from torchexp.stat import RunningAvgDict



class TrainInterface:
    def __init__(self, config, paras, id2accent):

        ### config setting
        self.config = config
        self.paras = paras
        self.train_type = 'evaluation'
        self.is_memmap = paras.is_memmap
        self.is_bucket = paras.is_bucket
        self.model_name = paras.model_name
        self.eval_ival = config['solver']['eval_ival']
        self.log_ival = config['solver']['log_ival']
        self.half_batch_ilen = config['solver']['half_batch_ilen'],
        self.dev_max_ilen = config['solver']['dev_max_ilen']
        self.spm = spmlib.SentencePieceProcessor()
        self.spm.Load(config['solver']['spm_model'])
        self.id2units = ['<blank>']
        with open(config['solver']['spm_mapping']) as fin:
            for line in fin.readlines():
                # print(line.rstrip().split(' ')[0])
                self.id2units.append(line.rstrip().split(' ')[0])

        # with open()

        self.save_verbose = paras.save_verbose
        #######################################################################

        ### Set path
        cur_path = Path.cwd()

        if paras.pretrain:
            assert paras.pretrain_suffix, \
            "You should specify pretrain model and the corresponding prefix"

            if paras.pretrain_model_path:
                self.pretrain_model_path = Path(paras.pretrain_model_path)
            else:
                assert paras.pretrain_setting and paras.pretrain_step > 0, "Should specify pretrain_setting"
                self.pretrain_model_path = Path(cur_path, LOG_DIR, 'pretrain', \
                                                paras.pretrain_setting, paras.algo, \
                                                paras.pretrain_suffix, \
                                                id2accent[paras.pretrain_tgt_accent],\
                                                str(paras.pretrain_runs), f"snapshot.step.{paras.pretrain_step}")

            assert self.pretrain_model_path.exists(), \
                f"Pretrain model path {self.pretrain_model_path} not exists"
            self.pretrain_module = config['solver']['pretrain_module']
        else:
            assert paras.pretrain_suffix is None and paras.algo == 'no', \
            f"Training from scratch shouldn't have meta-learner {paras.algo} and pretrain_suffix"
            paras.pretrain_suffix = paras.eval_suffix

        self.accent= id2accent[paras.accent]
        self.data_dir = Path(config['solver']['data_root'], self.accent)
        self.log_dir = Path(cur_path, LOG_DIR,self.train_type, \
                            config['solver']['setting'], paras.algo, \
                            paras.pretrain_suffix, paras.eval_suffix, \
                            self.accent, str(paras.runs))
        ########################################################################

        ### Resume mechanism
        if not paras.resume:
            if self.log_dir.exists():
                assert paras.overwrite, \
                    f"Path exists ({self.log_dir}). Use --overwrite or change suffix"
                # time.sleep(10)
                logger.warning('Overwrite existing directory')
                rmtree(self.log_dir)

            self.log_dir.mkdir(parents=True)
            self.train_info = RunningAvgDict(decay_rate=0.99)
            self.global_step = 1
            self.ep = 0

        else:
            self.resume_model_path = self.log_dir.joinpath('snapshot.latest')
            info_dict_path = self.log_dir.joinpath('info_dict.latest')
            self.optimizer_path = self.log_dir.joinpath('optimizer.latest')

            assert self.optimizer_path.exists(), \
                f"Optimizer state {self.optimizer_path} not exists..."

            with open(Path(self.log_dir, 'epoch'),'r') as f:
                self.ep = int(f.read().strip())
            with open(Path(self.log_dir, 'global_step'),'r') as f:
                self.global_step = int(f.read().strip())

            assert self.resume_model_path.exists(),\
                f"{self.resume_model_path} not exists..."
            assert info_dict_path.exists(),\
                f"Training info {info_dict_path} not exists..."

            with open(info_dict_path, 'rb') as fin:
                self.train_info = pickle.load(fin)
        self.dashboard = Dashboard(config, paras, self.log_dir, \
                                   self.train_type, paras.resume)

    def load_data(self):
        logger.notice(f"Loading data from {self.data_dir} with {self.paras.njobs} threads")

        self.id2ch = dict()
        self.ch2id = dict()
        with open(self.data_dir.resolve().parents[0].joinpath('valid_train_en_unigram150_units.txt')) as fin:
            for line in fin.readlines():
                ch, idx = line.split(' ')
                self.id2ch[int(idx)] = ch
                self.ch2id[ch] = int(idx)
        logger.log(f"Train units: {self.ch2id.keys()}")

        setattr(self, 'train_set', get_loader(
            self.data_dir.joinpath('train'), 
            batch_size=self.config['solver']['batch_size'],
            min_ilen = self.config['solver']['min_ilen'],
            max_ilen = self.config['solver']['max_ilen'],
            half_batch_ilen = self.config['solver']['half_batch_ilen'],
            # bucket_reverse=True,
            bucket_reverse=False,
            is_memmap = self.is_memmap,
            is_bucket = self.is_bucket,
            num_workers = self.paras.njobs,
            # shuffle=False, #debug
        ))
        setattr(self, 'dev_set', get_loader(
            self.data_dir.joinpath('dev'),
            batch_size = self.config['solver']['dev_batch_size'],
            is_memmap = self.is_memmap,
            is_bucket = False,
            shuffle = False,
            num_workers = self.paras.njobs,
        ))

    def write_log(self, k, v):
        with open(self.log_dir.joinpath(k),'a') as fout:
            print(f'{self.global_step} {v}', file=fout)

    def log_msg(self):
        if self.global_step % self.log_ival == 0:
            logger.log_info(self.train_info, prefix='train')
            self.dashboard.log_info('train', self.train_info)
