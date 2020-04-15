import torch
from torch import nn
import torch.nn.functional as F

class Speller(nn.Module):
    """
    :param int idim: enc_o_dim + dec_dim (if use context residual)
    :param int dec_dim: hidden dim for BLSTM
    :param int nlayers: # of layers of BLSTM
    :param int odim: # of output classes (default: # of grapheme + some tags) 
    """
    def __init__(self, idim, dec_dim, nlayers, odim):
        super(Speller, self).__init__()

        self.idim = idim 
        self.dec_dim = dec_dim
        self.nlayers = nlayers
        self.odim = odim
        
        self.state_list = list()
        self.cell_list = list()

        self.rnn0 = nn.LSTMCell(idim, dec_dim)

        for i in range(1,self.nlayers):
            rnn = nn.LSTM(dec_dim, dec_dim)
            setattr(self, f"rnn{i}", rnn)

    def init_rnn(self, context):
        batch_size = context.size(0)
        self.state_list = [context.new_zeros(batch_size, self.dec_dim)] * (self.nlayers)
        self.cell_list = [context.new_zeros(batch_size, self.dec_dim)] * (self.nlayers)

    @property
    def hidden_state(self):
        return [s.clone().detach().cpu() for s in self.state_list], [c.clone().detach().cpu() for c in self.cell_list]

    @hidden_state.setter
    def hidden_state(self, state):
        device = self.state_list[0].device
        self.state_list = [s.to(device) for s in state[0]]
        self.cell_list = [c.to(device) for c in state[1]]

    def forward(self, enc):
        self.state_list[0], self.cell_list[0] = self.rnn0(enc, (self.state_list[0], self.cell_list[0]))
        for i in range(1, self.nlayers):
            self.state_list[i], self.state_list[0] = getattr(self, f'rnn{i}')(self.state_list[i-1],(self.state_list[i],self.cell_list[i]))

        return self.state_list[-1]
