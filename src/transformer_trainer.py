import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from src.marcos import *
from src.model.transformer.mono_transformer import Transformer
from src.model.transformer.loss import cal_performance
from src.model.transformer.optimizer import TransformerOptimizer
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("Transformer Trainer Init...")

    class TransformerTrainer(cls):
        def __init__(self, config, paras, id2accent):
            super(TransformerTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = Transformer(self.id2ch, self.config['asr_model']).cuda()
            self.asr_opt = TransformerOptimizer(
                torch.optim.Adam(self.asr_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                self.config['asr_model']['optimizer_opt']['k'],
                self.config['asr_model']['encoder']['d_model'],
                self.config['asr_model']['optimizer_opt']['warmup_steps']
            )
            self.label_smoothing = self.config['solver']['label_smoothing']
            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            super().load_model()

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train):

            pred, gold = self.asr_model(x, ilens, ys, olens)
            loss, acc = cal_performance(pred, gold, self.label_smoothing)


            if train:
                info = { 'loss': loss.item(), 'acc': acc}
                if self.global_step % 500 == 0:
                    self.probe_model(pred[0], gold[0])
                self.asr_opt.zero_grad()
                loss.backward()

            else:
                wer = self.metric_observer.batch_cal_wer(pred.detach(), gold)
                info = { 'wer': wer, 'loss':loss.item() }

            return info

        def probe_model(self, pred, ys_out):
            self.metric_observer.cal_wer(torch.argmax(pred, dim=-1), ys_out, show=True)

    return TransformerTrainer(config, paras, id2accent)
