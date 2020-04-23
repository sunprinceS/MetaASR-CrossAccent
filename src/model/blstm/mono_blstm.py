import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math

from src.modules.encoder import BlstmEncoder, RNNP
from src.nets_utils import to_device, lecun_normal_init_parameters
from src.marcos import BLANK_SYMBOL
import src.monitor.logger as logger

# import tensorflow as tf

from IPython import embed

CTC_BEAM_RATIO = 1.5 # DO NOT CHANGE THIS, MAY CAUSE OOM
LOG_0 = float(np.finfo(np.float32).min)
LOG_1 = 0


class MonoBLSTM(nn.Module):
    def __init__(self, id2char, model_para):
        super(MonoBLSTM, self).__init__()

        # <blank>   [id2char...]         <sos>          <eos>
        # 0       1...len(id2char)  len(id2char)+1  len(id2char)+1
        # same as espnet

        self.idim = model_para['encoder']['idim']
        self.odim = len(id2char)
        self.sos_id = len(id2char) - 1
        self.eos_id = len(id2char) - 1
        enc_o_dim = model_para['encoder']['odim']

        # Construct model
        self.encoder = BlstmEncoder(**model_para['encoder'])
        self.head = nn.Linear(enc_o_dim, self.odim)

        self.init_parameters()

        # Initalization for beam decode
        '''
            params (dict):
                recog_beam_width (int): size of beam
                recog_length_penalty (float): length penalty
                recog_lm_weight (float): weight of LM score
                recog_lm_usage (str): rescoring or shallow_fusion
        '''
        self.beam_params = {
                'recog_length_penalty': 0,
                # 'recog_lm_weight':0,
                'recog_lm_usage': 'rescoring'
                }
        self.vocab = self.odim
        self.blank_id = id2char.index(BLANK_SYMBOL)
        # Don't know yet
        self.space = -1

    @property
    def device(self):
        return next(self.parameters()).device

    def greedy_decode(self, xs_pad, ilens):
        return self.forward(xs_pad, ilens)

    def enc_forward(self, xs_pad, ilens, probe_layer):
        assert xs_pad.size(0) == ilens.size(0), "Batch size mismatch"
        batch_size = xs_pad.size(0)

        xs_pad = to_device(self,xs_pad)
        ilens = to_device(self, ilens)

        enc, enc_lens = self.encoder.enc_forward(xs_pad, ilens, probe_layer)
        return  enc, enc_lens

    def forward(self, xs_pad, ilens):
        """
        xs_pad: (B, Tmax, 83)  #83 is 80-dim fbank + 3-dim pitch
        ilens: torch.Tensor with size B
        """
        assert xs_pad.size(0) == ilens.size(0), "Batch size mismatch"
        batch_size = xs_pad.size(0)

        # Put data to device
        xs_pad = to_device(self,xs_pad)
        ilens = to_device(self,ilens)

        enc, enc_lens, _, _ = self.encoder(xs_pad, ilens)
        out = self.head(enc)

        return out, enc_lens

    def init_parameters(self):
        self.init_encoder()
        lecun_normal_init_parameters(self.head)

    def init_encoder(self):
        lecun_normal_init_parameters(self.encoder)

    def set_beam_decode_params(self, beam_size, lm_weight, lm_model_path):
        self.decode_beam_size = beam_size
        self.lm_weight = lm_weight
        self._set_lm_model(lm_model_path)
        
    def _set_lm_model(self, lm_model_path):
        if lm_model_path is None :
            self.rescore_lm = None
        else:
            try:
                import kenlm
            except:
                raise ValueError('Should install kenlm for decoding rescoring')
            self.rescore_lm = kenlm.Model(lm_model_path)

    def beam_decode(self, x, ilen):
        assert x.size(0) == 1, "Batch size should be 1 in beam_decode"

        x = to_device(self, x)
        ilen = to_device(self, ilen)

        enc, enc_len, _, _ = self.encoder(x, ilen)
        out = self.head(enc)

        #beam_decode_ans = self._beam_search(out, enc_len, decode_beam_size)
        beam_decode_ans = self.tf_beam_decode(out, enc_len, decode_beam_size, nbest=decode_beam_size)
        #embed()

        return beam_decode_ans, enc_len

    # Reference: https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py?fbclid=IwAR1k65lhp05AKU0Twbw0Oduv6HIXhdfiG_d46RASKY5iLyA7Os_F4m7E36w#L162
    def _beam_search(self, outputs, lengths, beam_size, lm=None, nbest=1):
        """Beam search decoding.
        Args:
            outputs (FloatTensor): `[B, T, enc_n_units]`
            lengths (list): A list of length `[B]`
            lm (RNNLM or GatedConvLM or TransformerLM):
            nbest (int):
        Returns:
            best_hyps (list): Best path hypothesis. `[B, L]`
        """
        bs = outputs.shape[0]
        params = self.beam_params
        beam_width = beam_size
        lp_weight = params['recog_length_penalty']
        lm_weight = self.lm_weight
        lm_usage = params['recog_lm_usage']

        best_hyps = []
        log_probs = F.log_softmax(outputs, dim=-1)

        for b in range(bs):
            # Elements in the beam are (prefix, (p_b, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank (in log space).
            beam = [{'hyp_id': [self.eos_id],  # <eos> is used for LM
                     'p_b': LOG_1,
                     'p_nb': LOG_0,
                     'score_lm': LOG_1,
                     'lmstate': None,
                     }]

            for t in range(lengths[b]):
                new_beam = []

                # Pick up the top-k scores
                log_probs_topk, topk_ids = torch.topk(
                    log_probs[b:b + 1, t], k=min(beam_width, self.vocab), dim=-1, largest=True, sorted=True)

                for i_beam in range(len(beam)):
                    hyp_id = beam[i_beam]['hyp_id'][:]
                    p_b = beam[i_beam]['p_b']
                    p_nb = beam[i_beam]['p_nb']
                    score_lm = beam[i_beam]['score_lm']

                    # case 1. hyp is not extended
                    new_p_b = np.logaddexp(p_b + log_probs[b, t, self.blank_id].item(),
                                           p_nb + log_probs[b, t, self.blank_id].item())
                    if len(hyp_id) > 1:
                        new_p_nb = p_nb + log_probs[b, t, hyp_id[-1]].item()
                    else:
                        new_p_nb = LOG_0
                    score_ctc = np.logaddexp(new_p_b, new_p_nb)
                    score_lp = len(hyp_id[1:]) * lp_weight
                    new_beam.append({'hyp_id': hyp_id,
                                     'score': score_ctc + score_lm + score_lp,
                                     'p_b': new_p_b,
                                     'p_nb': new_p_nb,
                                     'score_ctc': score_ctc,
                                     'score_lm': score_lm,
                                     'score_lp': score_lp,
                                     'lmstate': beam[i_beam]['lmstate'],
                                     })

                    # Update LM states for shallow fusion
                    if lm_weight > 0 and lm is not None and lm_usage == 'shallow_fusion':
                        _, lmstate, lm_log_probs = lm.predict(
                            eouts.new_zeros(1, 1).fill_(hyp_id[-1]), beam[i_beam]['lmstate'])
                    else:
                        lmstate = None

                    # case 2. hyp is extended
                    new_p_b = LOG_0
                    #for c in tensor2np(topk_ids)[0]:
                    for c in topk_ids.cpu().numpy()[0]:
                        p_t = log_probs[b, t, c].item()

                        if c == self.blank_id:
                            continue

                        c_prev = hyp_id[-1] if len(hyp_id) > 1 else None
                        if c == c_prev:
                            new_p_nb = p_b + p_t
                            # TODO(hirofumi): apply character LM here
                        else:
                            new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)
                            # TODO(hirofumi): apply character LM here
                            if c == self.space:
                                pass
                                # TODO(hirofumi): apply word LM here

                        score_ctc = np.logaddexp(new_p_b, new_p_nb)
                        score_lp = (len(hyp_id[1:]) + 1) * lp_weight
                        if lm_weight > 0 and lm is not None and lm_usage == 'shallow_fusion':
                            local_score_lm = lm_log_probs[0, 0, c].item() * lm_weight
                            score_lm += local_score_lm
                        new_beam.append({'hyp_id': hyp_id + [c],
                                         'score': score_ctc + score_lm + score_lp,
                                         'p_b': new_p_b,
                                         'p_nb': new_p_nb,
                                         'score_ctc': score_ctc,
                                         'score_lm': score_lm,
                                         'score_lp': score_lp,
                                         'lmstate': lmstate,
                                         })

                # Pruning
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_width]

            # Rescoing lattice
            if lm_weight > 0 and lm is not None and lm_usage == 'rescoring':
                new_beam = []
                for i_beam in range(len(beam)):
                    ys = [np2tensor(np.fromiter(beam[i_beam]['hyp_id'], dtype=np.int64), self.device_id)]
                    ys_pad = pad_list(ys, lm.pad)
                    _, _, lm_log_probs = lm.predict(ys_pad, None)
                    score_ctc = np.logaddexp(beam[i_beam]['p_b'], beam[i_beam]['p_nb'])
                    score_lm = lm_log_probs.sum() * lm_weight
                    score_lp = len(beam[i_beam]['hyp_id'][1:]) * lp_weight
                    new_beam.append({'hyp_id': beam[i_beam]['hyp_id'],
                                     'score': score_ctc + score_lm + score_lp,
                                     'score_ctc': score_ctc,
                                     'score_lp': score_lp,
                                     'score_lm': score_lm})
                beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)

            best_hyps.append(np.array(beam[0]['hyp_id'][1:]))


        return np.array(best_hyps)

    def tf_beam_decode(self, outputs, lengths, beam_size, lm=None, nbest=1):
        def to_tf_format(tensors):
            # (B, T, C)
            # Originall blank_id = 0 --> move to C-1
            minus_inf = float(np.finfo(np.float32).min)
            fake = np.concatenate([tensors, tensors[:, :, 0:1]], axis=-1)
            fake[:, :, 0] = minus_inf
            return fake

        def array_to_str(array):
            # because begin and end are <bos> <eos>, not put them for rescoring
            if len(array) > 0 and array[0] == self.sos_id:
                array = array[1:]
            if len(array) > 0 and array[-1] == self.eos_id:
                array = array[:-1]
            return ' '.join(map(str, array))
        outputs = outputs.cpu().numpy()
        outputs = to_tf_format(outputs)
        outputs = tf.convert_to_tensor(outputs) # B, T, D
        outputs = tf.transpose(outputs, perm=[1, 0, 2]) #[max_time, batch_size, num_classes]

        lengths = tf.convert_to_tensor(lengths.numpy())
        lengths = tf.dtypes.cast(lengths, tf.int32)

        tops, prob = tf.nn.ctc_beam_search_decoder(outputs, lengths,
                beam_width=beam_size, top_paths=nbest)
        # tops = [d.values.numpy() for d in tops]
        tops = [d.values.eval(session=tf.Session()) for d in tops]
        choices = []
        for array in tops:
            string = array_to_str(array)
            if string != '':
                choices.append((array, self.rescore_lm.score(string)))
        #tuples = [(array, self.rescore_lm.score(array_to_str(array))) for array in tops]
        #embed()
        ans = sorted(choices, key=lambda x: x[1])[-1][0]

        return [ans]
