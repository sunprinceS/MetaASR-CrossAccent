import math
import pickle
import torch
from pathlib import Path
from abc import abstractmethod
from torch import nn
from torchexp.stat import RunningAvgDict
from functools import partial

from src.marcos import *
from src.nets_utils import to_device
from src.pretrain_interface import PretrainInterface
from src.model.transformer_pytorch.optimizer import TransformerOptimizer
from src.utils import get_bar
import src.monitor.logger as logger

class MultiASRInterface(PretrainInterface):
    
    def __init__(self, config, paras, id2accent):
        super(MultiASRInterface, self).__init__(config, paras, id2accent)

        self.asr_model = None
        self.asr_opt = None

        self.max_step = paras.max_step if paras.max_step > 0 else config['solver']['total_steps']
        self.dashboard.set_status('pretraining')

        self._train = partial(self.run_batch, train=True)
        self._eval = partial(self.run_batch, train=False)

    #NOTE: It's not so obvious to pick the best model for pretraining
    # (since we need to do evaluation offline actually)
    def save_best_model(self, tpe='wer', only_stat=False):
        assert self.asr_model is not None

        if not only_stat:
            model_save_path = self.log_dir.joinpath(f'model.{tpe}.best')
            logger.notice('Current best {}: {:3f}, save model to {}'.format(
                tpe.upper(), getattr(self,f'best_{tpe}'), model_save_path))

            torch.save(self.asr_model.state_dict(), model_save_path)

        with open(self.log_dir.joinpath(f'best_{tpe}'),'w') as fout:
            print('{} {}'.format(self.global_step, \
                                 getattr(self,f'best_{tpe}')), file=fout)

    def save_per_steps(self):
        assert self.asr_model is not None

        logger.log("Save model snapshot.latest", prefix='info')

        torch.save(self.asr_model.state_dict(), \
                   self.log_dir.joinpath("snapshot.latest"))
        with open(self.log_dir.joinpath("info_dict.latest"),'wb') as fin:
            pickle.dump(self.train_info, fin)

        with open(self.log_dir.joinpath("global_step"),'w') as fout:
            print(self.global_step, file=fout)

        # Used for transfer (as init weight for training)
        model_save_name = f"snapshot.step.{self.global_step}"
        logger.log(f"Save model {model_save_name}", prefix='info')

        torch.save(self.asr_model.state_dict(), self.log_dir.joinpath(model_save_name))
        self.dashboard.log_step()

    def load_model(self):
        logger.log("MultiASR model for pretraining initialization")

        if self.paras.resume:
            logger.notice(f"Resume pretraining from {self.global_step}")
            self.asr_model.load_state_dict(torch.load(self.resume_model_path))
            self.dashboard.set_step(self.global_step)
        else:
            logger.notice(f"Start pretraining from {self.global_step}")

    def write_tr_logs(self):
        for k, v in self.train_info.items():
            self.write_log(f"train_{k}", float(v))

    #TODO: seems silly
    def write_dev_logs(self, prefix, info_ls):
        for k, v in info_ls.items():
            self.write_log(f"{prefix}_{k}", float(v))

    def check_evaluate(self):
        if self.global_step % self.eval_ival == 0:
            logger.flush()
            self.asr_opt.zero_grad()
            self.evaluate()

    def train(self):
        try:
            while self.global_step < self.max_step:
                tbar = get_bar(total=self.eval_ival, \
                               desc=f"Step {self.global_step}", leave=True)

                for _ in range(self.eval_ival):
                    #TODO: we can add sampling method to compare Meta and Multi fair
                    idx, (x, ilens, ys, olens) = self.data_container.get_item()[0]

                    batch_size = len(ys)
                    info = self._train(idx, x, ilens, ys, olens, accent_idx = idx)
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

                    if self.global_step % self.save_ival == 0:
                        self.save_per_steps()
                    self.dashboard.check()
                tbar.close()

        except KeyboardInterrupt:
            logger.warning("Pretraining stopped")
            self.save_per_steps()
            self.dashboard.set_status('pretrained(SIGINT)')
        else:
            logger.notice("Pretraining completed")
            self.dashboard.set_status('pretrained')

    def evaluate(self):
        self.asr_model.eval()
        self.write_tr_logs()

        dev_info_ls = [RunningAvgDict(decay_rate=1.) for _ in range(self.num_pretrain)]
        for idx, dev_loader in enumerate(self.data_container.dev_loaders):
            tbar = get_bar(total=len(dev_loader), desc=f"Eval on {self.accents[idx]} @ step {self.global_step}")
            with torch.no_grad():
                for cur_b, (x, ilens, ys, olens) in enumerate(dev_loader):
                    
                    if ilens.max() > self.dev_max_ilen:
                        tbar.update(1)
                        continue
                    
                    batch_size = len(ys)
                    info = self._eval(idx, x, ilens, ys, olens)
                    dev_info_ls[idx].add(info, batch_size)

                    if cur_b % self.log_ival == 0:
                        logger.log_info(dev_info_ls[idx], prefix='test')

                    del x, ilens, ys, olens
                    tbar.update(1)

                logger.flush()
                tbar.close()

                self.dashboard.log_info(f"dev_{self.accents[idx]}", dev_info_ls[idx])
                self.write_dev_logs(f"dev_{self.accents[idx]}", dev_info_ls[idx])

        dev_avg_info = RunningAvgDict(decay_rate=1.0)
        for dev_info in dev_info_ls:
            dev_avg_info.add({k: float(v) for k, v in dev_info.items()})

        self.dashboard.log_info("dev", dev_avg_info)
        self.write_dev_logs("dev_avg", dev_avg_info)
        cur_cer = float(dev_avg_info['cer'])
        cur_wer = float(dev_avg_info['wer'])
        if cur_wer < self.best_wer:
            self.best_wer = cur_wer
            self.save_best_model()
        if cur_cer < self.best_cer:
            self.best_cer = cur_cer
            self.save_best_model('cer', only_stat=True)

        self.asr_model.train()

    @abstractmethod
    def run_batch(self, idx, x, ilens, ys, olens, train):
        raise NotImplementedError

