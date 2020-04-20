import torch.nn as nn

from src.model.transformer.attention import MultiHeadAttention
from src.model.transformer.module import PositionalEncoding, PositionwiseFeedForward
from src.model.transformer.utils import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    # def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 # d_model, d_inner, dropout=0.1, pe_maxlen=5000):
    def __init__(self, model_para, enc_in_dim, pe_maxlen = 5000):
        super(Encoder, self).__init__()
        self.d_model = model_para['encoder']['d_model']
        self.n_layers = model_para['encoder']['nlayers']
        self.n_head = model_para['encoder']['nhead']
        self.d_k = self.d_model
        self.d_v = self.d_model
        self.d_inner = model_para['encoder']['d_inner']
        self.dropout_rate = model_para['encoder']['dropout']
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(enc_in_dim, self.d_model)
        self.layer_norm_in = nn.LayerNorm(self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=self.dropout_rate)
            for _ in range(self.n_layers)])

    def forward(self, enc_in_pad, enc_lens, return_attns=False):
        """
        Args:
            padded_input: B * T * vgg_o_dim
            input_lengths: B

        Returns:
            enc_output: B x T x ???
        """
        enc_slf_attn_list = []

        # Prepare masks
        # if True: means non-padding (OPPOSITE as my implementation)
        non_pad_mask = get_non_pad_mask(enc_in_pad, input_lengths=enc_lens) # B * T * 1

        max_len = enc_in_pad.size(1)
        slf_attn_mask = get_attn_pad_mask(enc_in_pad, enc_lens, max_len)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(enc_in_pad)) +
            self.positional_encoding(enc_in_pad))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output, None


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
