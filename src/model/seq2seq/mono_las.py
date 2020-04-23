import torch 
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
import math

from src.marcos import *
from src.nets_utils import to_device, init_weights, init_gate
from src.modules.encoder import VGG
import src.monitor.logger as logger




class MonoLAS(nn.Module):
    def __init__(self, id2char, model_para):
        super(MonoLAS, self).__init__()

        # <sos>   [id2char...]  <eos>

        self.idim = model_para['encoder']['idim']
        #TODO: should we make sos_id and eos_id the same??? if not, ???
        self.odim = len(id2char)
        self.sos_id = len(id2char) - 1
        self.eos_id = len(id2char) - 1
        enc_o_dim = model_para['encoder']['odim']
        dec_dim = model_para['decoder']['dec_dim']

        # Construct model
        self.vgg = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        # vgg_o_dim = np.ceil(np.array(idim, dtype=np.float32) / 2)
        # vgg_o_dim = np.ceil(np.array(vgg_o_dim, dtype=np.float32) / 2)
        # vgg_o_dim = int(vgg_o_dim) * 256
        # self.vgg_o_dim = vgg_o_dim
        # self.blstm = RNNP(vgg_o_dim, self.nlayers, enc_dim, proj_dim, odim, self.sample_rate, self.dropout)

        self.encoder = Listener(**model_para['encoder'])
        self.attention = AttLoc(enc_dim=enc_o_dim, dec_dim=dec_dim, **model_para['attention'])
        self.embed = nn.Embedding(self.odim, dec_dim)
        self.char_trans = nn.Linear(dec_dim, self.odim)
        self.decoder = Speller(idim=enc_o_dim+dec_dim, odim=self.odim, **model_para['decoder'])

        self.init_parameters()


    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, xs_pad, ilens, ys_pad, decode_step, tf=False, tf_rate=0.0, vis_att=False):
        """
        xs_pad: (B, Tmax, 83)  #83 is 80-dim fbank + 3-dim pitch
        ilens: torch.Tensor with size B
        ys_pad: (B, Lmax), will parse in list of tensors in ctc
        """
        # NOTE: in each forward, we at most visualize one utterance attention
        # map (since I want to compare different length effect), and choose the
        # longest utterance in this batch for simplicity

        assert xs_pad.size(0) == ilens.size(0) == ys_pad.size(0), "Batch size mismatch"
        batch_size = xs_pad.size(0)

        # Put data to device
        xs_pad = to_device(self,xs_pad)
        ilens = to_device(self,ilens)
        ys_pad = to_device(self, ys_pad)

        enc, enc_lens, _ = self.encoder(xs_pad, ilens)
        ctc_output = self.ctc_layer(enc)

        sos = ys_pad.new([self.sos_id])
        eos = ys_pad.new([self.eos_id])

        if tf:
            # ys_pad_in = torch.cat((sos.repeat(batch_size).view(batch_size,-1),ys_pad),1)
            # teacher = self.embed(ys_pad_in) #(B, L+1, dec_dim)
            teacher = self.embed(ys_pad) #(B, L, dec_dim)

        self.decoder.init_rnn(enc)
        self.attention.reset()

        last_char_emb = self.embed(sos.repeat(batch_size))# B * dec_dim

        output_char_seq = list()
        if vis_att:
            output_att_seq = list()

        att_w = None

        for t in range(decode_step):
            att_c, att_w = self.attention(enc, enc_lens, self.decoder.state_list[0], att_w)

            dec_inp = torch.cat([last_char_emb, att_c], dim=-1) # (B, dec_dim + enc_o_dim)
            dec_out = self.decoder(dec_inp) #(B, dec_dim)
            cur_char = self.char_trans(dec_out) #(B, odim)

            # Teacher forcing
            if tf and t < decode_step - 1:
                # if random.random() < tf_rate:
                if torch.rand(1).item() < tf_rate:
                    last_char_emb = teacher[:,t,:]
                else: # scheduled sampling
                    sampled_char = Categorical(F.softmax(cur_char,dim=-1)).sample()
                    last_char_emb = self.embed(sampled_char)
            else: # greedy pick
                last_char_emb = self.embed(torch.argmax(cur_char,dim=-1))

            output_char_seq.append(cur_char)
            if vis_att:
                output_att_seq.append(att_w.detach()[0].view(enc_lens[0],1))

        att_output = torch.stack(output_char_seq, dim=1).view(batch_size*decode_step,-1)
        # att_output = torch.stack(output_char_seq, dim=1) # (B,T,o_dim)
        att_map = torch.stack(output_att_seq, dim=1) if vis_att else None
        # att_map = torch.stack(output_att_seq,dim=1) # (T,L)

        return ctc_output, enc_lens, att_output, att_map

    def init_parameters(self):
        self.apply(init_weights)

        for l in range(self.decoder.layer):
            bias = getattr(self.decoder.layers, f"bias_ih_l{l}")
            bias = init_gate(bias)
