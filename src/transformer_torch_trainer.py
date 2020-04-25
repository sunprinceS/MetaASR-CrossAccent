import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from src.marcos import *
from src.model.transformer_pytorch.mono_transformer_torch import MyTransformer
import torch_optimizer as optim
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("Transformer Trainer Init...")

    class TransformerTrainer(cls):
        def __init__(self, config, paras, id2accent):
            super(TransformerTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = MyTransformer(self.id2ch, self.config['asr_model']).cuda()
            self.asr_opt = optim.RAdam(self.asr_model.parameters(), betas=(0.9, 0.98), eps=1e-9)
            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            super().load_model()

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train):

            batch_size = len(ys)
            pred, gold = self.asr_model(x, ilens, ys, olens)
            pred_cat = pred.view(-1, pred.size(2))
            gold_cat = gold.contiguous().view(-1)
            loss = F.cross_entropy(pred_cat, gold_cat, ignore_index=IGNORE_ID)

            pred_cat = pred_cat.detach().max(1)[1]
            non_pad_mask = gold_cat.ne(IGNORE_ID)
            n_correct = pred_cat.eq(gold_cat).masked_select(non_pad_mask).sum().item()
            n_total = non_pad_mask.sum().item()

            if train:
                info = { 'loss': loss.item(), 'acc': float(n_correct)/n_total}
                # if self.global_step % 5 == 0:
                if self.global_step % 500 == 0:
                    self.probe_model(pred.detach(), gold)
                self.asr_opt.zero_grad()
                loss.backward()

            else:
                cer = self.metric_observer.batch_cal_er(pred.detach(), gold, ['att'], ['cer'])['att_cer']
                wer = self.metric_observer.batch_cal_er(pred.detach(), gold, ['att'], ['wer'])['att_wer']
                info = { 'cer': cer, 'wer': wer, 'loss':loss.item(), 'acc': float(n_correct)/n_total}

            return info

        def probe_model(self, pred, ys_out):
            self.metric_observer.cal_att_cer(torch.argmax(pred[0], dim=-1), ys_out[0], show=True, show_decode=True)
            self.metric_observer.cal_att_wer(torch.argmax(pred[0], dim=-1), ys_out[0], show=True)

    return TransformerTrainer(config, paras, id2accent)
