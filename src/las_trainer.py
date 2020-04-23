import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from src.marcos import *
from src.model.seq2seq.mono_las import MonoLAS
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("LAS Trainer Inint...")

    class LASTrainer(cls):
        def __init__(self, config, paras, id2accent):
            super(LASTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = MonoLAS(self.id2ch, self.config['asr_model']).cuda()
            self.seq_loss = nn.CrossEntropyLoss(ignore_index = IGNORE_ID, reduction='none')
            
            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            assert self.config['asr_model']['optimizer']['type'] == 'adadelta', "Use AdaDelta for pure seq2seq"
            self.asr_opt = getattr(torch.optim,\
                                   self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(),\
                                        **self.config['asr_model']['optimizer_opt'])

            # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                # self.asr_opt, mode='min', 
                                                # factor=0.2, patience=3, 
                                                # verbose=True)
            super().load_model()

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train):
            sos = ys[0].new([self.sos_id])
            eos = ys[0].new([self.eos_id])
            ys_out = [torch.cat]

            if train:
                info = {'loss':loss.item(), 'acc':acc}
                # if self.global_step % 5 == 0:
                if self.global_step % 500 == 0:
                    self.probe_model(pred, ys)
                self.asr_opt.zero_grad()
                loss.backward()
            else:
                wer = self.metric_observer.batch_cal_wer(pred.detach(), ys, ['att'])['att']
                info = {'wer': wer, 'loss': loss.item(), 'acc':acc}

            return info

        def probe_model(self, pred, ys_out):
            self.metric_observer.cal_att_wer(torch.argmax(pred[0],dim=-1),ys[0], show=True)

    return LASTrainer(config, paras, id2accent)
