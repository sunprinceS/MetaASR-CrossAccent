import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
from math import floor
from src.model.transformer.encoder import Encoder
from src.model.transformer.deocoder import Decoder

from src.nets_utils import to_device, lecun_normal_init_parameters, make_bool_pad_mask
import src.monitor.logger as logger

class Transformer(nn.Module):

    def __init__(self, id2char, model_para):
        super(Transformer, self).__init__()

        self.idim = model_para['encoder']['idim']

        #FIXME: need to remove these hardcoded thing later
        self.odim = len(id2char) + 2
        self.sos_id = len(id2char) + 1
        self.eos_id = len(id2char) + 1
        self.blank_id = 0
        self.space_id = -1 #FIXME: what is this
        self.vgg_ch_dim = 128
        
        self.feat_extractor = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.vgg_ch_dim,self.vgg_ch_dim, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
        )
        self.vgg_o_dim = self.vgg_ch_dim * floor(self.idim/4)


        self.encoder = Encoder(model_para, self.vgg_o_dim)
        self.decoder = Decoder(sos_id, eos_id, id2char, model_para)

        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_feature(self, xs_pad, ilens):
        batch_size = xs_pad.size(0)

        xs_pad = to_device(self, xs_pad)
        xs_pad = xs_pad.view(batch_size, xs_pad.size(1), 1, xs_pad.size(2)).transpose(1,2) # B * 1 * T * D(83)
        xs_pad = self.feat_extractor(xs_pad) # B * T * D' (128 * 20)
        ilens = torch.floor(ilens.to(dtype=torch.float32)/4).to(dtype=torch.int64)
        xs_pad = xs_pad.transpose(1,2)
        xs_pad = xs_pad.contiguous().view(batch_size,  xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        xs_pad = self.vgg2enc(xs_pad) # B * T * d_model
        
        return xs_pad, ilens

    def forward(self, xs_pad, ilens, ys_pad, olens):
        assert xs_pad.size(0) == ilens.size(0) == len(ys_pad) == olens.size(0), "Batch size mismatch"
        xs_pad, enc_lens = self.extract_feature(xs_pad, ilens)

        enc_out_pad, *_ = self.encoder(enc_in_pad, enc_lens)

        # pred is score before softmax
        pred, gold, *_ = self.decoder(ys_pad, enc_out_pad, enc_lens)

        return pred, gold


        # enc_outs_pad, enc_lens, *_ = self.encoder(xs_pad, ilens)
