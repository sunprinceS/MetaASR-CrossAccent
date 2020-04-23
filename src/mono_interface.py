import math
import pickle
import torch
from pathlib import Path
from abc import abstractmethod
from torch import nn
from torchexp.stat import RunningAvgDict
from functools import partial
from collections import OrderedDict

from src.marcos import *
from src.train_interface import TrainInterface
from src.model.transformer.optimizer import TransformerOptimizer
from src.utils import get_bar
import src.monitor.logger as logger


class MonoASRInterface(TrainInterface):

    def __init__(self, config, paras, id2accent):
        super(MonoASRInterface, self).__init__(config, paras, id2accent)
        # need implemented in children class
        self.asr_model = None
        self.asr_opt = None
        self.lr_scheduler = None
        
        self.max_epoch = config['solver']['total_epochs']
        self.dashboard.set_status('training')

        self._train= partial(self.run_batch, train=True)
        self._eval= partial(self.run_batch, train=False)

    def save_per_epoch(self):

        if self.save_verbose:
            logger.log(f"Save model snapshot.ep.{self.ep}")
            torch.save(self.asr_model.state_dict(), \
                       self.log_dir.joinpath(f"snapshot.ep.{self.ep}"))

        logger.log("Save model snapshot.latest", prefix='info')
        torch.save(self.asr_model.state_dict(), \
                   self.log_dir.joinpath("snapshot.latest"))
        if isinstance(self.asr_opt, TransformerOptimizer):
            with open(self.log_dir.joinpath("optimizer.latest"), "wb") as fout:
                pickle.dump(self.asr_opt, fout)
        else:
            torch.save(self.asr_opt.state_dict(), \
                       self.log_dir.joinpath("optimizer.latest"))

        with open(self.log_dir.joinpath("info_dict.latest"),'wb') as fout:
            pickle.dump(self.train_info, fout)

        with open(self.log_dir.joinpath("global_step"),'w') as fout:
            print(self.global_step, file=fout)
        self.dashboard.log_step()

        with open(Path(self.log_dir,'epoch'),'w') as fout:
            print(self.ep, file=fout)

    #TODO: move to basic_trainer
    def save_best_model(self, tpe='wer'):
        assert self.asr_model is not None
        model_save_path = self.log_dir.joinpath(f'model.{tpe}.best')
        logger.notice('Current best {}: {:3f}, save model to {}'.format(
            tpe.upper(), getattr(self,f'best_{tpe}'), model_save_path))

        torch.save(self.asr_model.state_dict(), model_save_path)
        with open(self.log_dir.joinpath(f'best_{tpe}'),'w') as fout:
            print('{} {}'.format(self.global_step, \
                                 getattr(self,f'best_{tpe}')), file=fout)

    def filter_model(self,state_dict):
        ret_state_dict = OrderedDict()
        for k, v in state_dict.items():
            module_name = k.split('.')[0]
            if module_name in self.pretrain_module:
                ret_state_dict[k] = v
        return ret_state_dict

    def load_model(self):
        logger.log("ASR model initialization")

        if self.paras.resume:
            logger.notice(f"Resume training from epoch {self.ep} (best wer: {self.best_wer})")
            self.asr_model.load_state_dict(torch.load(self.resume_model_path))
            if isinstance(self.asr_opt, TransformerOptimizer):
                with open(self.optimizer_path, 'rb') as fin:
                    self.asr_opt = pickle.load(fin)
            else:
                self.asr_opt.load_state_dict(torch.load(self.optimizer_path))
            self.dashboard.set_step(self.global_step)
        elif self.paras.pretrain:
            model_dict = self.asr_model.state_dict()
            logger.notice(f"Load pretraining {','.join(self.pretrain_module)} from {self.pretrain_model_path}")
            pretrain_dict = self.filter_model(torch.load(self.pretrain_model_path))
            model_dict.update(pretrain_dict)
            self.asr_model.load_state_dict(model_dict)
            logger.notice("Done!")
        else: # simple monolingual training from step 0
            logger.notice("Training from scratch")


    def check_evaluate(self):
        if self.global_step % self.eval_ival == 0:
            logger.flush()
            self.asr_opt.zero_grad()
            self.evaluate()

    def save_init(self):
            logger.log(f"Save model snapshot.init", prefix='info')
            torch.save(self.asr_model.state_dict(), \
                       self.log_dir.joinpath(f"snapshot.init"))

    def train(self):
        # self.evaluate()
        try:
            if self.save_verbose:
                self.save_init()
            while self.ep < self.max_epoch:
                tbar = get_bar(total=len(self.train_set), \
                               desc=f"Epoch {self.ep}", leave=True)

                for cur_b, (x, ilens, ys, olens) in enumerate(self.train_set):

                    batch_size = len(ys)
                    info = self._train(cur_b, x, ilens, ys, olens)
                    self.train_info.add(info, batch_size)

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.asr_model.parameters(), GRAD_CLIP)

                    if math.isnan(grad_norm):
                        logger.warning(f"grad norm NaN @ step {self.global_step}")
                    else:
                        self.asr_opt.step()

                    if isinstance(self.asr_opt, TransformerOptimizer):
                        self.log_msg(self.asr_opt.lr)
                    else:
                        self.log_msg()
                    self.check_evaluate()

                    self.global_step += 1
                    self.dashboard.step()

                    del x, ilens, ys, olens
                    tbar.update(1)

                self.ep += 1
                self.save_per_epoch()
                self.dashboard.check()
                tbar.close()

        except KeyboardInterrupt:
            logger.warning("Training stopped")
            self.evaluate()
            self.dashboard.set_status('trained(SIGINT)')
        else:
            logger.notice("Training completed")
            self.dashboard.set_status('trained')

    def evaluate(self):
        self.asr_model.eval()

        dev_info = RunningAvgDict(decay_rate=1.)
        tbar = get_bar(total = len(self.dev_set), 
                       desc=f"Eval @step{self.global_step}", leave=True)
        
        with torch.no_grad():
            for cur_b, (x, ilens, ys, olens) in enumerate(self.dev_set):

                if ilens.max() > self.dev_max_ilen:
                    tbar.update(1)
                    continue

                batch_size = len(ys)
                info = self._eval(cur_b, x, ilens, ys, olens)
                dev_info.add(info, batch_size)

                if cur_b % self.log_ival == 0:
                    logger.log_info(dev_info, prefix='test')

                del x, ilens, ys, olens
                tbar.update(1)

            logger.flush()
            tbar.close()

            self.dashboard.log_info('dev', dev_info)
            self.write_logs(dev_info)

            cur_wer = float(dev_info['wer'])
            if cur_wer < self.best_wer:
                self.best_wer = cur_wer
                self.save_best_model()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(float(dev_info['loss']))

        self.asr_model.train()

    @abstractmethod
    def run_batch(self, cur_b, x, ilens, ys, olens, train):
        raise NotImplementedError
