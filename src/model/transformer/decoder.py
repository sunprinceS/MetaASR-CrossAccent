import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets_utils import to_device
from src.model.transformer.attention import MultiHeadAttention
from src.model.transformer.module import PositionalEncoding, PositionwiseFeedForward
from src.model.transformer.utils import (IGNORE_ID, get_attn_key_pad_mask, get_attn_pad_mask,
                                           get_non_pad_mask, get_subsequent_mask, pad_list)


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    # def __init__(
            # self, sos_id, eos_id,
            # n_tgt_vocab, d_word_vec,
            # n_layers, n_head, d_k, d_v,
            # d_model, d_inner, dropout=0.1,
            # tgt_emb_prj_weight_sharing=True,
            # pe_maxlen=5000):
    def __init__(self, model_para, sos_id, eos_id, n_tgt_vocab, tgt_emb_prj_weight_sharing=True, pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = model_para['decoder']['d_model']
        self.n_layers = model_para['decoder']['nlayers']
        self.n_head = model_para['decoder']['nhead']
        self.d_model = model_para['decoder']['d_model']
        self.d_k = self.d_model
        self.d_v = self.d_model
        self.d_inner = model_para['decoder']['d_inner']
        self.dropout_rate = model_para['decoder']['dropout']
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(self.n_tgt_vocab, self.d_word_vec)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, dropout=self.dropout_rate)
            for _ in range(self.n_layers)])

        self.tgt_word_prj = nn.Linear(self.d_model, n_tgt_vocab, bias=False)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if self.tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (self.d_model ** -0.5)
        else:
            self.x_logit_scale = 1.
        
    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess(self, ys):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        ys: list, len B
        olens: B
        """
        # ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        sos = ys[0].new([self.sos_id])
        eos = ys[0].new([self.eos_id])

        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        ys_in_pad = to_device(self, ys_in_pad)
        ys_out_pad = to_device(self, ys_out_pad)

        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, enc_pad, enc_lens, ys, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        if return_attns:
            dec_slf_attn_list, dec_enc_attn_list = [], []

        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(ys)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(ys_in_pad, pad_idx=self.eos_id)

        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     pad_idx=self.eos_id)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # Assume we know the ground truth length
        max_olen = ys_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(enc_pad, enc_lens, max_olen)

        # Forward
        dec_output = self.dropout(self.tgt_word_emb(ys_in_pad) * self.x_logit_scale + self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_pad,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        seq_logit = self.tgt_word_prj(dec_output)

        # Return
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold, None, None


    # def recognize_beam(self, encoder_outputs, char_list, args):
        # """Beam search, decode one utterence now.
        # Args:
            # encoder_outputs: T x H
            # char_list: list of character
            # args: args.beam

        # Returns:
            # nbest_hyps:
        # """
        # # search params
        # beam = args.beam_size
        # nbest = args.nbest
        # if args.decode_max_len == 0:
            # maxlen = encoder_outputs.size(0)
        # else:
            # maxlen = args.decode_max_len

        # encoder_outputs = encoder_outputs.unsqueeze(0)

        # # prepare sos
        # ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # # yseq: 1xT
        # hyp = {'score': 0.0, 'yseq': ys}
        # hyps = [hyp]
        # ended_hyps = []

        # for i in range(maxlen):
            # hyps_best_kept = []
            # for hyp in hyps:
                # ys = hyp['yseq']  # 1 x i

                # # -- Prepare masks
                # non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                # slf_attn_mask = get_subsequent_mask(ys)

                # # -- Forward
                # dec_output = self.dropout(
                    # self.tgt_word_emb(ys) * self.x_logit_scale +
                    # self.positional_encoding(ys))

                # for dec_layer in self.layer_stack:
                    # dec_output, _, _ = dec_layer(
                        # dec_output, encoder_outputs,
                        # non_pad_mask=non_pad_mask,
                        # slf_attn_mask=slf_attn_mask,
                        # dec_enc_attn_mask=None)

                # seq_logit = self.tgt_word_prj(dec_output[:, -1])

                # local_scores = F.log_softmax(seq_logit, dim=1)
                # # topk scores
                # local_best_scores, local_best_ids = torch.topk(
                    # local_scores, beam, dim=1)

                # for j in range(beam):
                    # new_hyp = {}
                    # new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    # new_hyp['yseq'] = torch.ones(1, (1+ys.size(1))).type_as(encoder_outputs).long()
                    # new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    # new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # # will be (2 x beam) hyps at most
                    # hyps_best_kept.append(new_hyp)


                # hyps_best_kept = sorted(hyps_best_kept,
                                        # key=lambda x: x['score'],
                                        # reverse=True)[:beam]
            # # end for hyp in hyps
            # hyps = hyps_best_kept

            # # add eos in the final loop to avoid that there are no ended hyps
            # if i == maxlen - 1:
                # for hyp in hyps:
                    # hyp['yseq'] = torch.cat([hyp['yseq'],
                                             # torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()], dim=1)

            # # add ended hypothes to a final list, and removed them from current hypothes
            # # (this will be a probmlem, number of hyps < beam)
            # remained_hyps = []
            # for hyp in hyps:
                # if hyp['yseq'][0, -1] == self.eos_id:
                    # ended_hyps.append(hyp)
                # else:
                    # remained_hyps.append(hyp)

            # hyps = remained_hyps
            # if len(hyps) > 0:
                # print('remeined hypothes: ' + str(len(hyps)))
            # else:
                # print('no hypothesis. Finish decoding.')
                # break

            # for hyp in hyps:
                # print('hypo: ' + ''.join([char_list[int(x)]
                                          # for x in hyp['yseq'][0, 1:]]))
        # # end for i in range(maxlen)
        # nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            # :min(len(ended_hyps), nbest)]
        # # compitable with LAS implementation
        # for hyp in nbest_hyps:
            # hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        # return nbest_hyps


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
