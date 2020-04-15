import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from src.nets_utils import make_pad_mask, to_device

def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states

class Listener(nn.Module):
    """
    :param int idim: dim of acoustic feature, default: 83 (fbank + pitch)
    :param int enc_dim: hidden dim for BLSTM
    :param int proj_dim: projection dim for BLSTMP
    :param int odim: encoder output dimension (usually set as proj_dim)
    :param str sample_rate: e.g.  1_2_2_1_1
    :param str dropout: e.g. 0_0_0_0_0
    """

    def __init__(self, idim, enc_dim, proj_dim, odim, sample_rate, dropout):
        super(Listener, self).__init__()
        # Setting
        self.idim = idim
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.odim = odim
        self.sample_rate = [int(v) for v in sample_rate.split('_')]
        self.dropout = [float(v) for v in dropout.split('_')]

        # Parameters checking
        assert len(self.sample_rate) == len(self.dropout), 'Number of layer mismatch'
        self.nlayers = len(self.sample_rate)
        assert self.nlayers>=1,'Listener should have at least 1 layer'

        # VGG Extractor
        self.vgg = VGGExtractor(idim, in_channel=1)
        rnn_idim = self.vgg.odim
        self.blstm = RNNP(rnn_idim, self.nlayers, enc_dim, proj_dim, odim, self.sample_rate, self.dropout)

    def forward(self, xs_pad, ilens, prev_state=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, 83)
        :param torch.IntTensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)

        :return: batch of hidden state sequences (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        out, enc_lens = self.vgg(xs_pad, ilens)
        out, enc_lens, cur_state = self.blstm(out, enc_lens, prev_state)
        mask = to_device(self, make_pad_mask(enc_lens).unsqueeze(-1))

        return out.masked_fill(mask, .0), enc_lens, cur_state


class RNNP(nn.Module):
    """RNN with projection layer module

    :param int idim: dimension of inputs, default: odim of VGGExtractor
    :param int nlayers: Number of layers
    :param int enc_dim: hidden dim of BLSTMP
    :param int proj_dim: proj. dim of BLSTMP
    :param int odim: ouput of encoder dimension (default: proj_dim)
    :param list of int sample_rate: e.g [1,2,2,1,1]
    :param list of float dropout: e.g [.0,.0,.0,.0,.0]
    """

    def __init__(self, idim, nlayers, enc_dim, proj_dim, odim, sample_rate, dropout):
        super(RNNP, self).__init__()
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.odim = odim
        self.nlayers = nlayers
        self.sample_rate = sample_rate
        self.dropout = dropout

        self.rnn0 = nn.LSTM(idim, enc_dim, dropout=dropout[0], num_layers=1, bidirectional=True, batch_first=True)
        self.bt0 = nn.Linear(2 * enc_dim, proj_dim)


        for i in range(1,self.nlayers):
            rnn = nn.LSTM(proj_dim, enc_dim, dropout=dropout[i], num_layers=1, bidirectional=True, batch_first=True)
            setattr(self, f"rnn{i}", rnn)

            if i == self.nlayers - 1:
                setattr(self, f"bt{i}", nn.Linear(2 * enc_dim, odim))
            else:
                setattr(self, f"bt{i}", nn.Linear(2 * enc_dim, proj_dim))

    def enc_forward(self, xs_pad, enc_lens, probe_layer, prev_state=None):
        hid_states = []

        assert probe_layer <= self.nlayers,\
            f"Probe layer ({probe_layer}) can't exceed # of layers ({self.nlayers})"

        for i in range(probe_layer):
            xs_pack = pack_padded_sequence(xs_pad, enc_lens, batch_first = True)
            rnn = getattr(self, f"rnn{i}")
            rnn.flatten_parameters()
            if prev_state is not None:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(xs_pack, hx=None if prev_state is None else prev_state[i])
            hid_states.append(states)

            ys_pad, enc_lens= pad_packed_sequence(ys, batch_first=True)
            # ys_pad: (B, T, enc_dim)

            sub = self.sample_rate[i]
            if sub > 1:
                ys_pad = ys_pad[:,::sub]
                enc_lens = torch.LongTensor([int(i+1) // sub for i in enc_lens])
            projected = getattr(self, f"bt{i}")(ys_pad.contiguous().view(-1, ys_pad.size(2))) #(B*T, proj_dim)
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, to_device(self,enc_lens), hid_states

    def forward(self, xs_pad, enc_lens, prev_state=None):
        """RNNP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, vgg_o_dim)
        :param list of int enc_lens
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        hid_states = []

        for i in range(self.nlayers):
            xs_pack = pack_padded_sequence(xs_pad, enc_lens, batch_first = True)
            rnn = getattr(self, f"rnn{i}")
            rnn.flatten_parameters()
            if prev_state is not None:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(xs_pack, hx=None if prev_state is None else prev_state[i])
            hid_states.append(states)

            ys_pad, enc_lens= pad_packed_sequence(ys, batch_first=True)
            # ys_pad: (B, T, enc_dim)

            sub = self.sample_rate[i]
            if sub > 1:
                ys_pad = ys_pad[:,::sub]
                enc_lens = torch.LongTensor([int(i+1) // sub for i in enc_lens])
            projected = getattr(self, f"bt{i}")(ys_pad.contiguous().view(-1, ys_pad.size(2))) #(B*T, proj_dim)
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, to_device(self,enc_lens), hid_states

class VGGExtractor(nn.Module):
    def __init__(self, idim, in_channel):
        super(VGGExtractor, self).__init__()
        self.in_channel = in_channel
        self.freq_dim, self.odim = self._get_vgg_dim(idim, in_channel)


        self.conv1_1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # Half-time dimension
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # Half-time dimension

    def _get_vgg_dim(self, idim, in_channel, output_channel=128):
        assert idim%in_channel == 0, "Input dimension should be the times of in_channel"
        freq_dim = idim // in_channel
        odim = idim / in_channel
        odim = np.ceil(np.array(odim,dtype=np.float32) / 2)
        odim = np.ceil(np.array(odim,dtype=np.float32) / 2)

        return freq_dim, int(odim) * output_channel

    def forward(self, xs_pad, ilens, **kwargs):
        """VGGExtractor Forward
        :param torch.Tensor xs_pad: B * Tmax * D
        :param torch.IntTensor ilens: batch of lengths of input sequence (size: B)
        :return batch of padded hidden state sequence (B, Tmax // 4, 128 * D //4)
        :rtype: torch.Tensor
        """

        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1,2)
        #xs_pad: B * C * T * D', where D' = D // C

        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = self.pool1(xs_pad)
        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = self.pool2(xs_pad)

        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)

        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        xs_pad = xs_pad.transpose(1,2)
        xs_pad = xs_pad.contiguous().view(xs_pad.size(0),xs_pad.size(1),xs_pad.size(2) * xs_pad.size(3))

        return xs_pad, ilens

class BlstmEncoder(nn.Module):
    """
    :param int idim: dim of acoustic feature, default: 83 (fbank + pitch)
    :param int enc_dim: hidden dim for BLSTM
    :param int proj_dim: projection dim for BLSTMP
    :param int odim: encoder output dimension (usually set as proj_dim)
    :param str sample_rate: e.g.  1_2_2_1_1
    :param str dropout: e.g. 0_0_0_0_0
    """

    def __init__(self, idim, enc_dim, proj_dim, odim, sample_rate, dropout):
        super(BlstmEncoder, self).__init__()
        self.idim = idim
        self.enc_dim = enc_dim
        self.proj_dim = proj_dim
        self.odim = odim
        self.sample_rate = [int(v) for v in sample_rate.split('_')]
        self.dropout = [float(v) for v in dropout.split('_')]

        # Parameters checking
        assert len(self.sample_rate) == len(self.dropout), 'Number of layer mismatch'
        self.nlayers = len(self.sample_rate)
        assert self.nlayers>=1,'BlstmEncoder should have at least 1 layer'

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
        vgg_o_dim = np.ceil(np.array(idim, dtype=np.float32) / 2)
        vgg_o_dim = np.ceil(np.array(vgg_o_dim, dtype=np.float32) / 2)
        vgg_o_dim = int(vgg_o_dim) * 256
        self.vgg_o_dim = vgg_o_dim
        self.blstm = RNNP(vgg_o_dim, self.nlayers, enc_dim, proj_dim, odim, self.sample_rate, self.dropout)

    def enc_forward(self, xs_pad, ilens, probe_layer):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), 1, xs_pad.size(2)).transpose(1,2)
        xs_pad = self.vgg(xs_pad)

        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        enc_lens = np.array(np.ceil(ilens/2), dtype=np.int64)
        enc_lens = np.array(np.ceil(np.array(enc_lens, dtype=np.float32)/2), dtype=np.int64).tolist()

        xs_pad = xs_pad.transpose(1,2)
        xs_pad = xs_pad.contiguous().view(xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))

        if probe_layer == 0: # return VGG embedding
            mask = to_device(self, make_pad_mask(enc_lens).unsqueeze(-1))
            return xs_pad.masked_fill(mask,.0), enc_lens

        out, enc_lens, cur_state = self.blstm.enc_forward(xs_pad, enc_lens, probe_layer)

        mask = to_device(self, make_pad_mask(enc_lens).unsqueeze(-1))
        return out.masked_fill(mask, .0), enc_lens


    def forward(self, xs_pad, ilens, prev_state=None):
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), 1, xs_pad.size(2)).transpose(1,2)
        xs_pad = self.vgg(xs_pad)

        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        enc_lens = np.array(np.ceil(ilens/2), dtype=np.int64)
        enc_lens = np.array(np.ceil(np.array(enc_lens, dtype=np.float32)/2), dtype=np.int64).tolist()

        xs_pad = xs_pad.transpose(1,2)
        xs_pad = xs_pad.contiguous().view(xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        vgg_out = xs_pad.clone()

        out, enc_lens, cur_state = self.blstm(xs_pad, enc_lens, prev_state)
        mask = to_device(self, make_pad_mask(enc_lens).unsqueeze(-1))
        return out.masked_fill(mask, .0), enc_lens, cur_state, vgg_out
