import torch
import numpy as np
import editdistance
import sentencepiece as spmlib

import src.monitor.logger as logger
from itertools import groupby

class Metric:
    def __init__(self, model_path, id2units):
        self.spm = spmlib.SentencePieceProcessor()
        self.spm.Load(model_path)
        self.id2units = id2units
        self.sos_id = self.id2units.index('<s>')
        self.eos_id = self.id2units.index('</s>')

        logger.log(f"Train units: {self.id2units}")

    def batch_cal_wer(self, preds, ys):
        pred = torch.argmax(preds, dim=-1)
        batch_size = pred.size(0)

        wer = 0.0
        for h, y in zip(pred, ys):
            wer += self.cal_wer(h, y)

        return wer / batch_size

    def cal_wer(self, pred, y, show=False):
        show_pred = pred.tolist()
        show_pred = [x[0] for x in groupby(show_pred)]
        show_pred = [x for x in show_pred if x != self.sos_id and x!= self.eos_id]
        show_pred_text = self.spm.DecodePieces([self.id2units[x] for x in show_pred])
        
        show_y = y.tolist()
        show_y_text = self.spm.DecodePieces([self.id2units[x] for x in show_y if x!= self.sos_id and x != self.eos_id])

        wer = float(editdistance.eval(show_pred_text, show_y_text)) / len(show_y_text) * 100
        
        if show:
            logger.log(f"Hyp:\t {show_pred_text}", prefix='debug')
            logger.log(f"Ref:\t {show_y_text}", prefix='debug')
            logger.log(f"WER: {wer}", prefix='debug')

        return wer

        


# def cal_cer(preds, ys):
    # pred = torch.argmax(preds, dim=-1)
    # batch_size = pred.size(0)
    
    # cer = 0.0
    # for h,y in zip(pred, ys):
        # hh = h.tolist()
        # hh = [x[0] for x in groupby(hh)]
        # hh = [x for x in hh if x != 0] # remove blank
        # yy = y.tolist()

        # ed = editdistance.eval(hh, yy)
        # cer += (float(ed) / len(yy))

    # return 100 * cer / batch_size

def cal_att_cer(preds, ys, eos_id):
    assert len(ys) == preds.size(0), "Batch size of ys and preds conflict"
    batch_size = len(ys)

    pred = torch.argmax(preds, dim=-1) # B * decode_step
    preds = _get_preds(pred, eos_id) # list of preds (each at most decode_step)

    cer = 0.0
    for h, y in zip(pred, ys):
        hh = h.tolist()
        hh = [x[0] for x in groupby(hh)]
        hh = [x for x in hh if x != 0]
        yy = y.tolist()

        # logger.log(f"Hypothesis {hh}",prefix='debug')
        # logger.log(f"Truth {yy}", prefix='debug')
        ed = editdistance.eval(hh,yy)
        cer += (float(ed) / len(yy))
        # logger.log(f"Edit distance {(float(ed) / len(yy))}")
    return 100 * cer / batch_size
    
# def cal_error_rate(pred_pad, y_pad, batch_size, pred_max_len, olens, eos_id, ignore_idx=-1, cal_wer=False):
    # pred = torch.argmax(pred_pad, dim=-1).view(batch_size, pred_max_len)
    # preds = _get_preds(pred, eos_id)
    # mask = (y_pad != ignore_idx)
    # ys = _get_ys(y_pad.masked_select(mask), olens)

    # cer = 0.0
    # for h,y in zip(preds, ys):
        # hh = h.tolist()
        # yy = y.tolist()
        # hh = [x[0] for x in groupby(hh)]
        # ed = editdistance.eval(hh,yy)
        # cer += (float(ed) / len(yy))

    # return 100 * cer / batch_size

# def cal_acc(pred_pad, y_pad, ignore_idx=-1):
    # assert pred_pad.size(0) == y_pad.size(0)
    # mask = (y_pad != ignore_idx)

    # pred = torch.argmax(pred_pad, dim=-1).masked_select(mask)
    # y = y_pad.masked_select(mask)
    # numerator = torch.sum(pred == y)
    # denominator = torch.sum(mask)
    # return float(numerator) / float(denominator)

# def _get_preds(pred_pad, eos_id):
    # """
    # pred_pad: (B, L)
    # Return: list of lists
    # """
    # ret = []
    # for vec in pred_pad:
        # eos_loc = (vec == eos_id).nonzero()
        # if len(eos_loc) > 0:
            # stop_point = eos_loc[0][0].item()
            # ret.append(vec[:stop_point])
        # else:
            # ret.append(vec)
    # return ret

