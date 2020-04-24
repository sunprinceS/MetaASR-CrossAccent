import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from src.marcos import *
from src.model.blstm.mono_blstm import MonoBLSTM
from src.nets_utils import to_device
import src.monitor.logger as logger

def get_trainer(cls, config, paras, id2accent):
    logger.notice("BLSTM Trainer Init...")

    class BLSTMTrainer(cls):

        def __init__(self, config, paras, id2accent):
            super(BLSTMTrainer, self).__init__(config, paras, id2accent)

        def set_model(self):
            self.asr_model = MonoBLSTM(self.id2ch, self.config['asr_model']).cuda()
            self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            
            self.sos_id = self.asr_model.sos_id
            self.eos_id = self.asr_model.eos_id

            self.asr_opt = getattr(torch.optim, \
                                   self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), \
                                        **self.config['asr_model']['optimizer_opt'])

            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                self.asr_opt, mode='min', 
                                                factor=0.2, patience=3, 
                                                verbose=True)
            super().load_model()
            self.freeze_encoder(paras.freeze_layer)
            
        def freeze_encoder(self, module):
            if module is not None:
                if module == 'VGG':
                    for p in self.asr_model.encoder.vgg.parameters():
                        p.requires_grad = False
                elif module == 'VGG_BLSTM':
                    for p in self.asr_model.encoder.parameters():
                        p.requires_grad = False
                else:
                    raise ValueError(f"Unknown freeze layer {module} (VGG, VGG_BLSTM)")

                logger.log(f"Freeze {' '.join(module.split('_'))} layer", prefix='info')

        def exec(self):
            self.train()

        def run_batch(self, cur_b, x, ilens, ys, olens, train):
            sos = ys[0].new([self.sos_id])
            eos = ys[0].new([self.eos_id])
            ys_out = [torch.cat([sos, y, eos], dim=0) for y in ys]
            olens += 2 # pad <sos> and <eos>

            y_true = torch.cat(ys_out)

            pred, enc_lens = self.asr_model(x, ilens)
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
                    self.probe_model(pred, ys)
                self.asr_opt.zero_grad()
                loss.backward()

            else:
                cer = self.metric_observer.batch_cal_er(pred.detach(), ys, ['ctc'], ['cer'])['ctc_cer']
                wer = self.metric_observer.batch_cal_er(pred.detach(), ys, ['ctc'], ['wer'])['ctc_wer']
                info = { 'cer':cer, 'wer': wer, 'loss':loss.item() }

            return info

        def probe_model(self, pred, ys):
            self.metric_observer.cal_ctc_cer(torch.argmax(pred[0], dim=-1), ys[0], show=True, show_decode=True)
            self.metric_observer.cal_ctc_wer(torch.argmax(pred[0], dim=-1), ys[0], show=True)

    return BLSTMTrainer(config, paras, id2accent)
