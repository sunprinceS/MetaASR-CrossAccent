import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math
from math import floor

from src.nets_utils import to_device, lecun_normal_init_parameters, make_bool_pad_mask
import src.monitor.logger as logger

# TODO:
# add relative positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, model_para):
        self.sos_id = sos_id
        self.eos_id = eos_id

    def preprocess(self, ys_pad):
    def forward(self, ys_pad, enc_outs_pad, enc_lens, return_attn = False):
        dec_self_attn_ls, dec_enc_attn_ls = [], []

        ys_in_pad, ys_out_pad = self.preprocess(ys_pad)

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

        # self.vgg2enc = nn.Linear(self.vgg_o_dim, model_para['encoder']['d_model'])
        self.vgg2enc = nn.Linear(self.vgg_o_dim, model_para['encoder']['d_model'])
        self.pos_encoder = PositionalEncoding(model_para['encoder']['d_model'], model_para['encoder']['dropout'])
        encoder_layer = nn.TransformerEncoderLayer(model_para['encoder']['d_model'], #512
                                                   model_para['encoder']['nhead'], # 2
                                                   model_para['encoder']['dim_inner'], # 2048
                                                   model_para['encoder']['dropout'] # 0.1
                                                   ) 
        self.encoder = nn.TransformerEncoder(encoder_layer, model_para['encoder']['nlayers'])
        # self.decoder = xxx

        # self.init_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    # def greedy_decode(self, xs_pad, ilens):
        # return  self.forward(xs_pad, ilens)

    def forward(self, xs_pad, ilens, ys_pad, olens):
        enc_outs_pad, enc_lens = self.enc_forward(xs_pad, ilens, ys_pad, olens) 
        # enc_outs_pad: T * B * d_model
        # enc_lens: B
        

    def enc_forward(self, xs_pad, ilens, ys_pad, olens):

        assert xs_pad.size(0) == ilens.size(0) == len(ys_pad) == olens.size(0), "Batch size mismatch"
        batch_size = xs_pad.size(0)

        xs_pad = to_device(self,xs_pad)

        ## VGG forward
        xs_pad = xs_pad.view(batch_size, xs_pad.size(1), 1, xs_pad.size(2)).transpose(1,2) # B * 1 * T * D(83)
        xs_pad = self.feat_extractor(xs_pad) # B * T * D' (128 * 20)
        ilens = torch.floor(ilens.to(dtype=torch.float32)/4).to(dtype=torch.int64)
        xs_pad = xs_pad.transpose(1,2)
        xs_pad = xs_pad.contiguous().view(batch_size,  xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        xs_pad = self.vgg2enc(xs_pad) # B * T * d_model


        ## TransformerEncoder forward
        xs_pad = xs_pad.transpose(0,1) # T * B * d_model
        xs_pad = self.pos_encoder(xs_pad)
        pad_mask = make_bool_pad_mask(ilens) # T * B * d_model (if True: means padding)
        pad_mask = to_device(self, pad_mask)

        enc_pad = self.encoder(xs_pad, src_key_padding_mask=pad_mask) # T * B * d_model

        return enc_pad, ilens

    def dec_forward(self, enc_outs_pad, enc_lens, ys_pad, olens):
