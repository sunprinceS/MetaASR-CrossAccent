import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from src.marcos import *
from src.model.transformer_pytorch.mono_transformer_torch import MyTransformer
from src.model.transformer_pytorch.optimizer import TransformerOptimizer
import torch_optimizer as extra_optim
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("Transformer Trainer Init...")

    class TransformerTrainer(cls):
        def __init__(self, config, paras, id2accent):
            super(TransformerTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = MyTransformer(self.id2ch, self.config['asr_model']).cuda()
            self.label_smooth_rate = self.config['solver']['label_smoothing']
            if self.label_smooth_rate > 0.0:
                logger.log(f"Use label smoothing rate {self.label_smooth_rate}",prefix='info')
            # self.asr_opt = optim.RAdam(self.asr_model.parameters(), betas=(0.9, 0.98), eps=1e-9)
            # if self.config['asr_model']['optimizer_cls'] == 'noam':
            if 'inner_optimizer_cls' not in self.config['asr_model']: # multi or mono
                if self.config['asr_model']['optimizer_cls'] == 'noam':
                    logger.notice("Use noam optimizer, it is recommended to be used in mono-lingual training")
                    self.asr_opt = TransformerOptimizer(
                        torch.optim.Adam(self.asr_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                        self.config['asr_model']['optimizer_opt']['k'],
                        self.config['asr_model']['d_model'],
                        self.config['asr_model']['optimizer_opt']['warmup_steps']
                    )
                elif self.config['asr_model']['optimizer_cls'] == 'RAdam':
                    logger.notice(f"Use third-library {self.config['asr_model']['optimizer_cls']} optimizer")
                    self.asr_opt = getattr(extra_optim,\
                                           self.config['asr_model']['optimizer_cls'])
                    self.asr_opt = self.asr_opt(self.asr_model.parameters(), \
                                                **self.config['asr_model']['optimizer_opt'])
                else:
                    logger.notice(f"Use {self.config['asr_model']['optimizer_cls']} optimizer, it is recommended to be used in fine-tuning")
                    self.asr_opt = getattr(torch.optim,\
                                           self.config['asr_model']['optimizer_cls'])
                    self.asr_opt = self.asr_opt(self.asr_model.parameters(), \
                                                **self.config['asr_model']['optimizer_opt'])
            else:
                logger.notice("During meta-training, model optimizer will reset after running each task")

            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            super().load_model()

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train, accent_idx=None):
            # if accent_idx is not None -> monolingual training

            batch_size = len(ys)
            pred, gold = self.asr_model(x, ilens, ys, olens)
            pred_cat = pred.view(-1, pred.size(2))
            gold_cat = gold.contiguous().view(-1)
            non_pad_mask = gold_cat.ne(IGNORE_ID)
            n_total = non_pad_mask.sum().item()

            if self.label_smooth_rate > 0.0:
                eps = self.label_smooth_rate
                n_class = pred_cat.size(1)

                gold_for_scatter = gold_cat.ne(IGNORE_ID).long() * gold_cat
                one_hot = torch.zeros_like(pred_cat).scatter(1, gold_for_scatter.view(-1,1),1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
                log_prb = F.log_softmax(pred_cat, dim=-1)
                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum() / n_total

            else:
                loss = F.cross_entropy(pred_cat, gold_cat, ignore_index=IGNORE_ID)

            pred_cat = pred_cat.detach().max(1)[1]
            n_correct = pred_cat.eq(gold_cat).masked_select(non_pad_mask).sum().item()

            if train:
                info = { 'loss': loss.item(), 'acc': float(n_correct)/n_total}
                # if self.global_step % 5 == 0:
                if self.global_step % 500 == 0:
                    self.probe_model(pred.detach(), gold, accent_idx)
                self.asr_opt.zero_grad()
                loss.backward()

            else:
                cer = self.metric_observer.batch_cal_er(pred.detach(), gold, ['att'], ['cer'])['att_cer']
                wer = self.metric_observer.batch_cal_er(pred.detach(), gold, ['att'], ['wer'])['att_wer']
                info = { 'cer': cer, 'wer': wer, 'loss':loss.item(), 'acc': float(n_correct)/n_total}

            return info

        def probe_model(self, pred, ys_out, accent_idx):
            if accent_idx is not None:
                logger.log(f"Probe on {self.accents[accent_idx]}", prefix='debug')
            self.metric_observer.cal_att_cer(torch.argmax(pred[0], dim=-1), ys_out[0], show=True, show_decode=True)
            self.metric_observer.cal_att_wer(torch.argmax(pred[0], dim=-1), ys_out[0], show=True)

    return TransformerTrainer(config, paras, id2accent)
