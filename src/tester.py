import pickle
import copy
import time
from itertools import groupby

from shutil import rmtree
from joblib import Parallel, delayed
from pathlib import Path
import torch

from comet_ml import Experiment, ExistingExperiment
from tqdm import tqdm

from src.marcos import *
from src.io.dataset import get_loader
from src.utils import get_bar
import src.monitor.logger as logger

class Tester:
    def __init__(self, config, paras, id2accent):

        self.config = config
        self.paras = paras
        self.train_type = 'evaluation'
        self.is_memmap = paras.is_memmap
        self.model_name = paras.model_name

        self.njobs = paras.njobs

        if paras.algo == 'no' and paras.pretrain_suffix is None:
            paras.pretrain_suffix = paras.eval_suffix

        ### Set path
        cur_path = Path.cwd()
        self.data_dir = Path(config['solver']['data_root'], id2accent[paras.accent])
        self.log_dir = Path(cur_path, LOG_DIR, self.train_type, 
                            config['solver']['setting'], paras.algo, \
                            paras.pretrain_suffix, paras.eval_suffix, \
                            id2accent[paras.accent], str(paras.runs))
        self.model_path = Path(self.log_dir, paras.test_model)
        assert self.model_path.exists(), f"{self.model_path.as_posix()} not exists..."
        self.decode_dir = Path(self.log_dir, paras.decode_suffix)


        ### Decode
        self.decode_mode = paras.decode_mode
        self.beam_decode_param = config['solver']['beam_decode']
        self.batch_size = paras.decode_batch_size
        self.use_gpu = paras.cuda

        if paras.decode_mode == 'lm_beam':
            assert paras.lm_model_path is not None, "In LM Beam decode mode, lm_model_path should be specified"
            # assert self.model_name == 'blstm', "LM Beam decode is only supported in blstm model"
            self.lm_model_path  = paras.lm_model_path
        else :
            self.lm_model_path = None

        # if paras.decode_mode == 'greedy':
            # self._decode = self.greedy_decode
        # elif paras.decode_mode == 'beam' or paras.decode_mode == 'lm_beam':
            # self._decode = self.beam_decode
        # else :
            # raise NotImplementedError
        #####################################################################

        ### Resume Mechanism
        if not paras.resume:
            if self.decode_dir.exists():
                assert paras.overwrite, \
                    f"Path exists ({self.decode_dir}). Use --overwrite or change decode suffix"
                # time.sleep(10)
                logger.warning('Overwrite existing directory')
                rmtree(self.decode_dir)
            self.decode_dir.mkdir(parents=True)
            self.prev_decode_step = -1
        else:
            with open(Path(self.decode_dir,'best-hyp'),'r') as f:
                for i, l in enumerate(f):
                    pass
                self.prev_decode_step = i+1
            logger.notice(f"Decode resume from {self.prev_decode_step}")

        ### Comet
        with open(Path(self.log_dir,'exp_key'),'r') as f:
            exp_key = f.read().strip()
            comet_exp = ExistingExperiment(previous_experiment=exp_key,
                                           project_name=COMET_PROJECT_NAME,
                                           workspace=COMET_WORKSPACE,
                                           auto_output_logging=None,
                                           auto_metric_logging=None,
                                           display_summary_level=0,
                                           )
        comet_exp.log_other('status','decode')

    def load_data(self):
        logger.notice(f"Loading data from {self.data_dir}")
        if self.model_name == 'blstm':
            self.id2ch = [BLANK_SYMBOL]
        elif self.model_name == 'transformer':
            self.id2ch = [SOS_SYMBOL]
        else:
            raise NotImplementedError

        with open(self.config['solver']['spm_mapping']) as fin:
            for line in fin.readlines():
                self.id2ch.append(line.rstrip().split(' ')[0])
            self.id2ch.append(EOS_SYMBOL)
        logger.log(f"Train units: {self.id2ch}")

        setattr(self, 'eval_set', get_loader(
            self.data_dir.joinpath('test'),
            # self.data_dir.joinpath('dev'),
            batch_size = self.batch_size,
            half_batch_ilen = 512 if self.batch_size > 1 else None,
            is_memmap = self.is_memmap,
            is_bucket = False,
            shuffle = False,
            num_workers = 1,
        ))

    def exec(self):
        if self.decode_mode != 'greedy':
            logger.notice(f"Start decoding with beam search (with beam size: {self.config['solver']['beam_decode']['beam_size']})")
            raise NotImplementedError(f"{self.decode_mode} haven't supported yet")
            self._decode = self.beam_decode
        else:
            logger.notice("Start greedy decoding")
            if self.batch_size > 1:
                dev = 'gpu' if self.use_gpu else 'cpu'
                logger.log(f"Number of utterance batches to decode: {len(self.eval_set)}, decoding with {self.batch_size} batch_size using {dev}")
                self._decode = self.batch_greedy_decode
                self.njobs = 1
            else:
                logger.log(f"Number of utterances to decode: {len(self.eval_set)}, decoding with {self.njobs} threads using cpu")
                self._decode = self.greedy_decode

        if self.njobs > 1:
            try:
                _ = Parallel(n_jobs=self.njobs)(delayed(self._decode)(i, x, ilen, y, olen) for i, (x, ilen, y, olen) in enumerate(self.eval_set))

            #NOTE: cannot log comet here, since it cannot serialize
            except KeyboardInterrupt:
                logger.warning("Decoding stopped")
            else:
                logger.notice("Decoding done")
                # self.comet_exp.log_other('status','decoded')
        else:

            tbar = get_bar(total=len(self.eval_set), leave=True)

            for cur_b, (xs, ilens, ys, olens) in enumerate(self.eval_set):
                self.batch_greedy_decode(xs, ilens, ys, olens)
                tbar.update(1)


    def set_model(self):

        logger.notice(f"Load trained ASR model from {self.model_path}")
        if self.model_name == 'blstm':
            from src.model.blstm.mono_blstm import MonoBLSTM as ASRModel
        elif self.model_name == 'las':
            from src.model.seq2seq.mono_las import MonoLAS as ASRModel
        elif self.model_name == 'transformer':
            from src.model.transformer_pytorch.mono_transformer_torch import MyTransformer as ASRModel
        else:
            raise NotImplementedError
        self.asr_model = ASRModel(self.id2ch, self.config['asr_model'])

        if self.use_gpu:
            self.asr_model.load_state_dict(torch.load(self.model_path))
            self.asr_model = self.asr_model.cuda()
        else:
            self.asr_model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.asr_model.eval()
        logger.log(f'ASR model device {self.asr_model.device}', prefix='debug')
        self.sos_id = self.asr_model.sos_id
        self.eos_id = self.asr_model.eos_id
        self.blank_id = None if self.model_name != 'blstm' else self.asr_model.blank_id

        #FIXME: will merge it later
        if self.decode_mode != 'greedy':
            if self.model_name == 'blstm':
                raise NotImplementedError
            elif self.model_name == 'transformer':
                raise NotImplementedError
            else:
                raise NotImplementedError

    def trim(self,hyp):
        assert isinstance(hyp, list)
        if self.model_name == 'blstm':
            #FIXME: should we discard everyting after eos_id???
            return [i for i in hyp if i < self.eos_id]
        elif self.model_name == 'transformer':
            # step1: remove everyting after eos
            eos_pos = len(hyp) + 1
            ret = []
            if len(hyp) > 1:
                for pos in range(1, len(hyp)):
                    if hyp[pos] == self.eos_id:
                        eos_pos = pos
                        break
                return hyp[:eos_pos]
            else:
                return ret
        else:
            raise NotImplementedError

    #TODO: make the following more modularized
    def batch_greedy_decode(self, xs, ilens, ys, olens):
        # if self.model_name == 'transformer':
            # # logger.log(torch.max(olens) - torch.min(olens), prefix='test')
            # logger.log(ilens, prefix='test')
            # return True
        with torch.no_grad():
            if self.model_name == 'blstm':
                preds, _ = self.asr_model(xs, ilens)
                preds = torch.argmax(preds, dim=-1)
                for pred, y in zip(preds, ys):
                    pred = self.trim(pred.tolist())
                    pred = [x[0] for x in groupby(pred)]
                    if self.blank_id is not None:
                        pred = [x for x in pred if x!= self.blank_id]
                    self.write_hyp(y.tolist(), pred)
                del preds
            elif self.model_name == 'transformer':
                preds = self.asr_model.recog(xs, ilens)
                preds = preds.transpose(0,1)
                # preds, _  = self.asr_model(xs, ilens, ys, olens)
                # preds = torch.argmax(preds, dim=-1)
                for pred, y in zip(preds, ys):
                    pred = self.trim(pred.tolist())
                    self.write_hyp(y.tolist(), pred)
                del preds
            else:
                raise NotImplementedError(f"{self.model_name} doesn't support greedy decode batchwise")

        del xs, ilens, ys, olens
        return True


    def greedy_decode(self, cur_step, x, ilen, y, olen):
        if cur_step > self.prev_decode_step:
            if cur_step % 500 == 0:
                logger.log(f"Current step {cur_step}")
            with torch.no_grad():
                hyp , _ = self.asr_model.greedy_decode(x, ilen)
                hyp = self.trim(torch.argmax(hyp[0], dim=-1).tolist())
                hyp = [x[0] for x in groupby(hyp)]
                if self.blank_id is not None:
                    hyp = [x for x in hyp if x != self.blank_id]
                # del model
            self.write_hyp(y[0].tolist(),hyp)
            del hyp
        del x, ilen, y, olen
        return True

    def beam_decode(self, cur_step ,x, ilen, y, olen):
        if cur_step > self.prev_decode_step:
            if cur_step % 500 == 0:
                logger.log(f"Current step {cur_step}")
            with torch.no_grad():
                model = copy.deepcopy(self.asr_model)
                hyp = model.beam_decode(x, ilen)
                del model
            self.write_hyp(y[0],hyp[0])
            del hyp
        del x, ilen, y, olen
        return True

    def write_hyp(self, y, hyp):
        with open(Path(self.decode_dir, 'best-hyp'),'a') as fout:
            fout.write("{}\t{}\n".format(" ".join(str(i) for i in y)," ".join(str(i) for i in hyp)))
