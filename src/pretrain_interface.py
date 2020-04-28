import pickle
import time

from shutil import rmtree
from pathlib import Path

from tqdm import tqdm
from src.marcos import *
import src.monitor.logger as logger
from src.io.dataset import DataContainer 
from src.monitor.metric import Metric
from torchexp.stat import RunningAvgDict

class PretrainInterface:

    def __init__(self, config, paras, id2accent):

        ### config setting
        self.config = config
        self.paras = paras
        self.train_type = 'pretrain'
        self.is_memmap = paras.is_memmap
        self.is_bucket = paras.is_bucket
        self.model_name = paras.model_name
        self.eval_ival = config['solver']['eval_ival']
        self.log_ival = config['solver']['log_ival']
        self.save_ival = config['solver']['save_ival']
        self.half_batch_ilen = config['solver']['half_batch_ilen']
        self.dev_max_ilen = config['solver']['dev_max_ilen']

        self.best_cer = INIT_BEST_ER
        self.best_wer = INIT_BEST_ER


        if self.paras.model_name == 'transformer':
            self.id2units = [SOS_SYMBOL]
            with open(config['solver']['spm_mapping']) as fin:
                for line in fin.readlines():
                    self.id2units.append(line.rstrip().split(' ')[0])
            self.id2units.append(EOS_SYMBOL)
            self.metric_observer = Metric(config['solver']['spm_model'], self.id2units, 0, len(self.id2units)-1)
        elif self.paras.model_name == 'blstm':
            self.id2units = [BLANK_SYMBOL]
            with open(config['solver']['spm_mapping']) as fin:
                for line in fin.readlines():
                    self.id2units.append(line.rstrip().split(' ')[0])
            self.id2units.append(EOS_SYMBOL)
            self.metric_observer = Metric(config['solver']['spm_model'], self.id2units, len(self.id2units)-1, len(self.id2units)-1)
        else:
            raise ValueError(f"Unknown model name {self.paras.model_name}")

        self.accents = [id2accent[accent_id] for accent_id in paras.pretrain_accents]
        self.num_pretrain = paras.num_pretrain
        self.tgt_accent = id2accent[paras.tgt_accent]

        self.max_step = paras.max_step if paras.max_step > 0 else config['solver']['total_steps']
        #######################################################################

        ### Set path
        assert self.num_pretrain == len(self.accents),\
            f"num_pretrain is {self.num_pretrain}, but got {len(self.accents)} in pretrain_accents"

        cur_path = Path.cwd()
        self.data_dirs = [Path(config['solver']['data_root']).joinpath(accent) for accent in self.accents]

        self.log_dir = Path(cur_path, LOG_DIR, self.train_type,  
                            config['solver']['setting'], paras.algo, 
                            paras.pretrain_suffix, self.tgt_accent, str(paras.runs))

        if not paras.resume:
            if self.log_dir.exists():
                assert paras.overwrite, \
                    f"Path exists ({self.log_dir}). Use --overwrite or change suffix"
                # time.sleep(10)
                logger.warning('Overwriting existing directory')
                rmtree(self.log_dir)

            self.log_dir.mkdir(parents=True)
            self.train_info = RunningAvgDict(decay_rate=0.99)
            self.global_step = 0
        else:
            self.resume_model_path = self.log_dir.joinpath('snapshot.latest')
            info_dict_path = self.log_dir.joinpath('info_dict.latest')
            self.optimizer_path = self.log_dir.joinpath('optimizer.latest')
            

            assert self.optimizer_path.exists(), \
                f"Optimizer state {self.optimizer_path} not exists..."
            
            with open(Path(self.log_dir,'global_step'),'r') as f:
                self.global_step = int(f.read().strip())

            assert self.resume_model_path.exists(), \
                f"{self.resume_model_path} not exists..."
            assert info_dict_path.exists(), \
                f"PreTraining info {info_dict_path} not exists..."

            with open(info_dict_path, 'rb') as fin:
                self.train_info = pickle.load(fin)
        if paras.use_tensorboard:
            from src.monitor.tb_dashboard import Dashboard
            logger.warning("Use tensorboard instead of comet")
        else:
            from src.monitor.dashboard import Dashboard

        self.dashboard = Dashboard(config, paras, self.log_dir, \
                                   self.train_type, paras.resume)
    def load_data(self):
        self.id2ch = self.id2units
        self.data_container = DataContainer(
                                self.data_dirs, 
                                batch_size=self.config['solver']['batch_size'],
                                dev_batch_size=self.config['solver']['dev_batch_size'],
                                is_memmap = self.is_memmap,
                                is_bucket = self.is_bucket,
                                num_workers = self.paras.njobs,
                                min_ilen = self.config['solver']['min_ilen'],
                                max_ilen = self.config['solver']['max_ilen'],
                                half_batch_ilen = self.config['solver']['half_batch_ilen'],
                                )

    def write_log(self, k, v):
        with open(self.log_dir.joinpath(k),'a') as fout:
            print(f"{self.global_step} {v}", file=fout)

    def log_msg(self, lr=None):
        if self.global_step % self.log_ival == 0:
            logger.log_info(self.train_info, prefix='train')
            self.dashboard.log_info('train', self.train_info)

            if lr is not None:
                self.dashboard.log_other('lr', lr)
