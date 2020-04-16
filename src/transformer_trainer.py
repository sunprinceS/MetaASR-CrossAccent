import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from src.marcos import *
from src.model.transformer.mono_transformer import Transformer
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("Transformer Trainer Init...")

    class TransformerTrainer(cls):
        def __init__(self, config, paras, id2accent):
            super(TransformerTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = Transformer(self.id2ch, self.config['asr_model']).cuda()
            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            super().load_model()

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train):
            sos = ys[0].new([self.sos_id])
            eos = ys[0].new([self.eos_id])
            ys_out = [torch.cat([sos, y, eos], dim=0) for y in ys]
            olens += 2 # pad <sos> and <eos>

            y_true = torch.cat(ys_out)


            #FIXME: check which y should pass
            pred, enc_lens = self.asr_model(x, ilens, ys_out, olens)
            olens = to_device(self.asr_model, olens)
            pred = F.log_softmax(pred, dim=-1) # (T, o_dim) 

            loss = self.ctc_loss(pred.transpose(0,1).contiguous(),
                                 y_true.cuda().to(dtype=torch.long),
                                 enc_lens.cpu().to(dtype=torch.long),
                                 olens.cpu().to(dtype=torch.long))

            if train:
                info = { 'loss': loss.item() }
                # if self.global_step % 5 == 0:
                if self.global_step % 500 == 0:
                    self.probe_model(pred, ys_out)
                self.asr_opt.zero_grad()
                loss.backward()

            else:
                wer = self.metric_observer.batch_cal_wer(pred.detach(), ys_out)
                info = { 'wer': wer, 'loss':loss.item() }

            return info

        def probe_model(self, pred, ys_out):
            self.metric_observer.cal_wer(torch.argmax(pred[0], dim=-1), ys_out[0], show=True)

    return TransformerTrainer(config, paras, id2accent)
