import torch
from torch import nn
import torch.nn.functional as F

from src.nets_utils import make_pad_mask, to_device

class AttLoc(nn.Module):
    """location-aware attention

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int enc_dim: odim of encoder
    :param int dec_dim: hidden dim of decoder BLSTMP
    :param att_dim: dim. of attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param mode: str, 'loc'
    """

    def __init__(self, enc_dim, dec_dim, att_dim, aconv_chans, aconv_filts, mode, **kwargs):
        super(AttLoc, self).__init__()
        self.mode = mode
        self.mlp_enc = nn.Linear(enc_dim, att_dim) # psi
        self.mlp_dec = nn.Linear(dec_dim, att_dim, bias=False) #phi
        self.mlp_att = nn.Linear(aconv_chans, att_dim, bias=False)
        # self.loc_conv = nn.Conv1d(1, aconv_chans, kernel_size=2*aconv_filts+1, padding=aconv_filts, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1) #glimpse

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.att_dim = att_dim

        self.enc_h = None
        self.enc_len_max= None
        self.precomp_enc_h = None
        self.mask = None

    def reset(self):
        """reset states"""
        self.enc_h = None
        self.enc_len_max= None
        self.precomp_enc_h = None
        self.mask = None

    def forward(self, comp_listner_feat, enc_lens, dec_h, att_prev, scaling=2.0):
        """AttLoc forward

        :param torch.Tensor comp_listner_feat: padded encoder hidden state (B, Tmax, enc_o_dim)
        :param list of int enc_lens
        :param dec_h: decoder hidden state (B, dec_dim)
        :param att_prev: previous attention scores (att_w) (B,Tmax)
        :param float scaling: scaling parameter before applying softmax

        :return: attention weighted decoder state, context  (B, dec_dim)
        :rtype: torch.Tensor
        :return: attention score (B, Tmax)
        :rtype: torch.Tensor
        """
        batch_size = comp_listner_feat.size(0)

        # pre-compute all h outside the decoder loop
        if self.precomp_enc_h is None:
            self.enc_h = comp_listner_feat # (B, T, enc_dim)
            self.enc_len_max = self.enc_h.size(1)
            # Only calculate once
            self.precomp_enc_h = self.mlp_enc(self.enc_h) # (B, T, att_dim)

        dec_h = dec_h.view(batch_size, self.dec_dim)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = to_device(self, (~make_pad_mask(enc_lens)).float())
            # att_prev = att_prev / att_prev.sum(dim=1).unsqueeze(-1)
            att_prev = att_prev / enc_lens.float().unsqueeze(-1) # (B, T)

        att_conv = self.loc_conv(att_prev.view(batch_size,1,1, self.enc_len_max)) # (B, C, 1, T)
        att_conv = att_conv.squeeze(2).transpose(1,2) # (B, T, C)
        att_conv = self.mlp_att(att_conv) # (B, T, att_dim)
        dec_h = self.mlp_dec(dec_h).view(batch_size, 1, self.att_dim) # (B, 1, att_dim)
        e = self.gvec(torch.tanh(att_conv + self.precomp_enc_h + dec_h)).squeeze(2) # (B, T)

        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_lens))

        # masked out e according to mask
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1) #(B, T)
        c = torch.bmm(w.view(batch_size, 1, self.enc_len_max), self.enc_h).squeeze(1) # (B, enc_dim)

        return c, w
