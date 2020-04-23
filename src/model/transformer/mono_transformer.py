import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
from math import floor
from src.model.transformer.encoder import Encoder
from src.model.transformer.decoder import Decoder
from src.marcos import SOS_SYMBOL, EOS_SYMBOL

from src.nets_utils import to_device
import src.monitor.logger as logger

class Transformer(nn.Module):

    def __init__(self, id2char, model_para):
        super(Transformer, self).__init__()

        self.idim = model_para['encoder']['idim']

        self.odim = len(id2char)
        self.sos_id = id2char.index(SOS_SYMBOL)
        # FIXME: Maybe we need to set sos_id = eos_id ???
        self.eos_id = id2char.index(EOS_SYMBOL)
        self.vgg_ch_dim = 128
        
        self.feat_extractor = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.vgg_ch_dim, self.vgg_ch_dim, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
        )
        self.vgg_o_dim = self.vgg_ch_dim * floor(self.idim/4)


        self.encoder = Encoder(model_para, self.vgg_o_dim)
        #NOTE: no-blank anymore since no ctc
        self.decoder = Decoder(model_para, self.sos_id, self.eos_id, self.odim)
        # self.decoder = Decoder(sos_id, eos_id, id2char, model_para)

        self.init_parameters()

    def init_parameters(self):
        #TODO: Why just init outside??? @@ -> need to check
        # for p in self.feat_extractor.parameters():
            # if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
        # self.encoder.init_parameters()
        # self.decoder.init_parameters()
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
        
        return xs_pad, ilens

    def forward(self, xs_pad, ilens, ys, olens):
        assert xs_pad.size(0) == ilens.size(0) == len(ys) == olens.size(0), "Batch size mismatch"
        enc_pad, enc_lens = self.extract_feature(xs_pad, ilens)

        enc_out_pad, _ = self.encoder(enc_pad, enc_lens)

        # pred is score before softmax
        # print(ys)
        # print(type(ys))
        # print(type(ys[0]))
        pred, gold, *_ = self.decoder(enc_out_pad, enc_lens, ys)

        return pred, gold


