import math
import pickle
import torch
from pathlib import Path
from abc import abstractmethod
from torch import nn
from torchexp.stat import RunningAvgDict
from functools import partial
from random import shuffle

from src.marcos import *
from src.nets_utils import to_device, clone_state_dict
from src.pretrain_interface import PretrainInterface
from src.model.transformer_pytorch.optimizer import TransformerOptimizer
from src.utils import get_bar
import src.monitor.logger as logger

# TODO:
# 1. TransformerOptimizer (aka noam optimizer )is also needed during meta-train (or can we
# automatically learn the lr?)

class FOMetaASRInterface(PretrainInterface):

    def __init__(self, config, paras, id2accent):
        super(FOMetaASRInterface, self).__init__(config, paras, id2accent)

        assert paras.meta_k is not None
        self.meta_k = paras.meta_k
        if paras.meta_batch_size is None:
            logger.log("Meta batch_size not set...", prefix='info')
            self.meta_batch_size = self.num_pretrain

        self.asr_model = None
        self.asr_opt = None

        self.max_step = paras.max_step if paras.max_step > 0 else config['solver']['total_steps']
        self.dashboard.set_status('pretraining')

        self._train = partial(self.run_batch, train=True)
        self._eval = partial(self.run_batch, train=False)

        self.meta_batch_size = paras.meta_batch_size
        self._updates = None
        self._counter = 0

        d_model = config['asr_model']['d_model']
        opt_k = config['asr_model']['meta']['optimizer_opt']['k']
        warmup_steps = config['asr_model']['meta']['optimizer_opt']['warmup_steps']
        self.inner_lr = d_model ** (-0.5) * opt_k * ((warmup_steps) ** (-0.5))

        logger.notice("Meta Interface Information")
        logger.log(f"Sampling strategy: {self.sample_strategy}", prefix='info')
        logger.log(f"Meta batch size  : {self.meta_batch_size}", prefix='info')
        logger.log(f"# inner-loop step: {self.meta_k}", prefix='info')
        # Max. of lr in noam
        logger.log( "Inner loop lr    : {}".format(round(self.inner_lr, 5)),prefix='info')

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
        logger.log("First Order GBML ASRmodel for pretraining initialization")

        if self.paras.resume:
            logger.notice(f"Resume pretraining from {self.global_step}")
            self.asr_model.load_state_dict(torch.load(self.resume_model_path))
            self.dashboard.set_step(self.global_step)
        else:
            logger.notice(f"Start pretraining from {self.global_step}")

        self._original = clone_state_dict(self.asr_model.state_dict(keep_vars=True))
        params = [p for p in self._original.values() if getattr(p, 'requires_grad', False)]

        if self.config['asr_model']['meta_opt_cls'] == 'noam':
            self.meta_opt = TransformerOptimizer(
                torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09),
                self.config['asr_model']['meta']['optimizer_opt']['k'],
                self.config['asr_model']['d_model'],
                self.config['asr_model']['meta']['optimizer_opt']['warmup_steps']
            )
        else:
            raise NotImplementedError(f"Should use noam optimizer in outer loop transformer learning, but got {self.asr_model['meta_opt_cls']}")

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
            task_ids = list(range(self.num_pretrain))
            while self.global_step < self.max_step:
                tbar = get_bar(total=self.eval_ival, \
                               desc=f"Step {self.global_step}", leave=True)
                for _ in range(self.eval_ival):
                    shuffle(task_ids)

                    #FIXME: Here split to inner-train and inner-test (should observe whether the performance drops)
                    for accent_id in task_ids[:self.meta_batch_size]:
                        # inner-loop learn
                        tr_batches = self.data_container.get_item(accent_id, self.meta_k)
                        self.run_task(tr_batches)

                        # inner-loop test
                        val_batch = self.data_container.get_item(accent_id)[0]
                        batch_size = len(val_batch[1][2])
                        info = self._train(val_batch[0],*val_batch[1], accent_idx = val_batch[0])
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.asr_model.parameters(), GRAD_CLIP)

                        if math.isnan(grad_norm):
                            logger.warning(f"grad norm NaN @ step {self.global_step} on {self.accents[accent_id]}, ignore...")

                        self._partial_meta_update()
                        del val_batch
                        self.train_info.add(info, batch_size)

                    self._final_meta_update()

                    self.log_msg(self.meta_opt.lr)
                    self.check_evaluate()
                    self.global_step += 1
                    self.dashboard.step()
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


    def _partial_meta_update(self):

        if self._updates is None:
            self._updates = {}
            for n, p in self._original.items():
                if not getattr(p, 'requires_grad', False):
                    continue
                if p.size():
                    self._updates[n] = p.new(*p.size()).zero_()
                else:
                    self._updates[n] = p.clone().zero_()

        for n, p in self.asr_model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue
            if self.paras.algo == 'fomaml':
                self._updates[n].add_(p.grad.data)
            else:
                raise ValueError(f"Not support meta algo {self.paras.algo}")

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self._original.items():
            if n not in self._updates:
                continue
            p.grad = self._updates[n]

        #TODO: should clip gradients in meta-gradient?
        # if self.dashboard.global_step % 50 == 0:
            # for n,p in self._original.items():
                # self.dashboard.add_histogram(p.grad.cpu(), "Meta",n,"grad")
        # grad_norm = nn.utils.clip_grad_norm_(
            # self._original.parameters(), GRAD_CLIP)
        # if math.isnan(grad_norm):
            # logger.warning(f"Meta-grad norm NaN @ step {self.global_step}")
        # else:
        self.meta_opt.step()
        self.meta_opt.zero_grad()
        self._counter = 0
        self._updates = None

    def run_task(self, batches):
        self._counter += 1

        self.asr_model.load_state_dict(self._original)
        self.asr_model.train()
        self.asr_opt = getattr(torch.optim, \
                               self.config['asr_model']['inner_optimizer_cls'])
        #TODO: how to set lr in inner-loop?? following noam's lr?? 
        # Should warmup here???????? How about fine-tune??????

        self.asr_opt = self.asr_opt(self.asr_model.parameters(),
                                    lr = self.inner_lr,
                                    momentum = self.config['asr_model']['inner_optimizer_opt']['momentum'],
                                    nesterov = self.config['asr_model']['inner_optimizer_opt']['nesterov'])

        for cnt, (idx, (x, ilens, ys, olens)) in enumerate(batches):
            # info = self._train(idx, x, ilens, ys, olens)
            self._train(idx, x, ilens, ys, olens)

            grad_norm = nn.utils.clip_grad_norm_(
                self.asr_model.parameters(), GRAD_CLIP)

            if math.isnan(grad_norm):
                logger.warning(f"grad norm NaN @ step {self.global_step}")
            else:
                self.asr_opt.step()

            del x, ilens, ys, olens


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

