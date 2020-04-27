import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import math
from math import floor

from src.marcos import IGNORE_ID
from src.nets_utils import to_device, lecun_normal_init_parameters, make_bool_pad_mask, generate_square_subsequent_mask
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


class MyTransformer(nn.Module):

    def __init__(self, id2char, model_para):
        super(MyTransformer, self).__init__()

        self.idim = model_para['idim']

        #FIXME: need to remove these hardcoded thing later
        self.odim = len(id2char)
        #TODO: check whether we need to make sos_id and eos_id different or the same
        self.sos_id = 0
        self.eos_id = len(id2char)-1
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
        self.vgg2enc = nn.Linear(self.vgg_o_dim, model_para['d_model'])

        self.pos_encoder = PositionalEncoding(model_para['d_model'], model_para['pos_dropout'])

        self.char_trans = nn.Linear(model_para['d_model'], self.odim)
        self.pre_embed = nn.Embedding(self.odim, model_para['d_model'])
        if model_para['tgt_share_weight'] != 0:
            logger.warning("Tie weight of char_trans and embedding")
            self.char_trans.weight = self.pre_embed.weight

        self.d_model = model_para['d_model']
        self.nhead = model_para['nheads']
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.nhead,
            dim_feedforward = model_para['d_inner'],
            dropout=model_para['dropout']
        )
        encoder_norm = nn.LayerNorm(model_para['d_model'])
        self.encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = model_para['encoder']['nlayers'],
            norm = encoder_norm
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.d_model,
            nhead = self.nhead,
            dim_feedforward = model_para['d_inner'],
            dropout=model_para['dropout']
        )
        decoder_norm = nn.LayerNorm(model_para['d_model'])
        self.decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = model_para['decoder']['nlayers'],
            norm = decoder_norm
        )

        self.init_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def extract_feat(self, xs_pad, ilens):
        batch_size = len(ilens)
        xs_pad = to_device(self,xs_pad.view(xs_pad.size(0), xs_pad.size(1), 1, xs_pad.size(2)).transpose(1,2))
        enc_pad = self.feat_extractor(xs_pad) # B * T * D' (128 * 20)
        enc_lens = torch.floor(ilens.to(dtype=torch.float32)/4).to(dtype=torch.int64)
        enc_pad = enc_pad.transpose(1,2)
        enc_pad = enc_pad.contiguous().view(batch_size,  enc_pad.size(1), enc_pad.size(2) * enc_pad.size(3))
        enc_pad = self.vgg2enc(enc_pad) # B * T * d_model

        return enc_pad, enc_lens

    def preprocess(self, ys, olens):

        #TODO: should we set ys_out_pad to eos ot ignore_id
        sos = ys[0].new([self.sos_id])
        eos = ys[0].new([self.eos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_in_pad = to_device(self,pad_sequence(ys_in, padding_value=self.eos_id, batch_first=False)) # L*B
        ys_in_pad = self.pre_embed(ys_in_pad) # L * B * d_model
        #TODO: should we add logit_scale of sqrt(d_model) here??
        ys_in_pad = self.pos_encoder(ys_in_pad)

        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        ys_out_pad = to_device(self,pad_sequence(ys_out, padding_value=IGNORE_ID, batch_first=True)) # B*L
        # ys_out_pad = pad_sequence(ys, padding_value=self.eos_id, batch_first=True) # B * L

        olens += 1

        return ys_in_pad, ys_out_pad, olens

    def recog(self, xs_pad, ilens):
        assert xs_pad.size(0) == ilens.size(0), "Batch size mismatch"
        batch_size = xs_pad.size(0)

        enc_pad, enc_lens = self.extract_feat(xs_pad, ilens)
        enc_pad = enc_pad.transpose(0,1) # T * B * d_model
        enc_pad = self.pos_encoder(enc_pad)
        enc_pad_mask = make_bool_pad_mask(enc_lens) # T * B * d_model (if True: means padding)
        enc_pad_mask = to_device(self, enc_pad_mask)

        decode_maxlen = enc_lens.max().item()
        
        memory = self.encoder(enc_pad, src_key_padding_mask = enc_pad_mask)

        # TODO: should we sample during decoding, or just use the logit??
        # use sampled char then embed currently

        # Init
        sos = torch.ones(1, batch_size).fill_(self.sos_id).to(dtype=torch.int64)
        sos = to_device(self,sos) # 1 * B
        out = to_device(self, torch.tensor([], dtype=torch.int64))

        for decode_step in range(1,decode_maxlen+1):
            ys_in_pad = self.pre_embed(torch.cat([sos, out])) # (decode_step) * B * d_model
            ys_in_pad = self.pos_encoder(ys_in_pad)
            tgt_self_attn_mask = to_device(self, generate_square_subsequent_mask(ys_in_pad.size(0)))

            out = self.decoder(ys_in_pad, memory, 
                               tgt_mask = tgt_self_attn_mask,
                               memory_key_padding_mask = enc_pad_mask)
            out = self.char_trans(out)
            out = torch.argmax(out, dim=-1)

        return out # L * B

    def forward(self, xs_pad, ilens, ys, olens):
        assert xs_pad.size(0) == ilens.size(0) == len(ys) == olens.size(0), "Batch size mismatch"

        enc_pad, enc_lens = self.extract_feat(xs_pad, ilens)

        enc_pad = enc_pad.transpose(0,1) # T * B * d_model
        enc_pad = self.pos_encoder(enc_pad)

        # enc_pad_mask will be used in src_key_padding_mask
        # TODO: should we use enc_pad_mask for memory_key_padding_mask, ENABLE it now
        enc_pad_mask = make_bool_pad_mask(enc_lens) # T * B * d_model (if True: means padding)
        enc_pad_mask = to_device(self, enc_pad_mask)

        ys_in_pad, ys_out_pad, olens = self.preprocess(ys, olens)
        # ys_in_pad: L * B * d_model

        # tgt_self_attn_mask will be used in tgt_mask (subsequent mask)
        tgt_self_attn_mask = to_device(self, generate_square_subsequent_mask(ys_in_pad.size(0)))

        # tgt_pad_mask will be used in tgt_key_padding_mask
        tgt_pad_mask = to_device(self, make_bool_pad_mask(olens))

        memory = self.encoder(enc_pad, src_key_padding_mask = enc_pad_mask)
        out = self.decoder(ys_in_pad, memory, 
                           tgt_mask=tgt_self_attn_mask,
                           memory_key_padding_mask=enc_pad_mask)
        # out: L * B * d_model
        out = out.transpose(0,1) # B * L * d_model
        logit = self.char_trans(out) # B * L * odim

        return logit, ys_out_pad
