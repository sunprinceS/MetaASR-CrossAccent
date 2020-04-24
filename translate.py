#!/usr/bin/env python3
import argparse
import re
import yaml
import json
from pathlib import Path
import sentencepiece as spmlib
from comet_ml import ExistingExperiment

from src.marcos import *
from src.utils import run_cmd
import src.monitor.logger as logger
import editdistance


parser = argparse.ArgumentParser(description='Evaluate the decoded hypothesis via sclite')

parser.add_argument('--config', type=str, help='Path to config file', required=True)
parser.add_argument('--accent', choices=AVAIL_ACCENTS, required=True)
parser.add_argument('--algo', choices=['reptile','fomaml', 'multi', 'fomaml_fast','no'], required=True)
parser.add_argument('--model_name', required=True, choices=['blstm','las'])
parser.add_argument('--eval_suffix', default=None, type=str, help='Evaluation suffix', required=True)
parser.add_argument('--decode_mode', default='greedy', type=str, choices=['greedy','beam','lm_beam'])
parser.add_argument('--pretrain_suffix',type=str, help='Pretrain model suffix', nargs='?',const=None)
parser.add_argument('--runs', type=int, help='run suffix', nargs='?',const=0)
parser.add_argument('--decode_suffix', default=None, type=str)  # will remove later
paras = parser.parse_args()
decode_suffix = f"{paras.decode_mode}_decode_{paras.decode_suffix}" if paras.decode_suffix else f"{paras.decode_mode}_decode" 

# Load some prerequisites
with open(Path('data','accent-code.json'),'r') as fin:
    id2accent = json.load(fin)
config = yaml.safe_load(open(paras.config,'r'))
if paras.pretrain_suffix is None:
    if paras.algo == 'no':
        paras.pretrain_suffix = paras.eval_suffix
    else:
        assert False, "pretrain_suffix should be specified if using pretrained model"

# Write to corresponding directory and comet exp
cur_path = Path.cwd()
log_dir = Path(cur_path, LOG_DIR, 'evaluation', 
                config['solver']['setting'], paras.algo, \
                paras.pretrain_suffix, paras.eval_suffix, \
                id2accent[paras.accent], str(paras.runs))
decode_dir = log_dir.joinpath(decode_suffix)

with open(Path(log_dir,'exp_key'),'r') as f:
    exp_key = f.read().strip()
    comet_exp = ExistingExperiment(previous_experiment=exp_key,
                                   project_name=COMET_PROJECT_NAME, 
                                   workspace=COMET_WORKSPACE,
                                   auto_output_logging=None,
                                   auto_metric_logging=None,
                                   display_summary_level=0,
                                   )

## The following need to be modified to remove redundant code
spm = spmlib.SentencePieceProcessor()
spm.Load(config['solver']['spm_model'])
id2units = list()
if paras.model_name == 'blstm':
    id2units.append(BLANK_SYMBOL)
    with open(config['solver']['spm_mapping']) as fin:
        for line in fin.readlines():
            id2units.append(line.rstrip().split(' ')[0])
        id2units.append('</s>')
            
else:
    raise NotImplementedError(f"{paras.model_name} is not implemented")

def to_list(s):
    ret = list(map(int, s.split(' ')))
    ret = [id2units[i] for i in ret]
    return ret

### Cal CER ####################################################################
logger.notice("CER calculating...")
cer = 0.0
with open(Path(decode_dir, 'best-hyp'),'r') as hyp_ref_in:
    cnt = 0
    for line in hyp_ref_in.readlines():
        cnt += 1
        ref, hyp = line.rstrip().split('\t')
        ref = spm.DecodePieces(to_list(ref))
        hyp = spm.DecodePieces(to_list(hyp))
        cer += (editdistance.eval(ref, hyp) / len(ref) * 100)
    cer = cer  / cnt
logger.log(f"CER: {cer}", prefix='test')
comet_exp.log_other(f"cer({paras.decode_mode})",round(cer,2))
with open(Path(decode_dir,'cer'),'w') as fout:
    print(str(cer), file=fout)
################################################################################

### Cal SER ####################################################################
logger.notice("Symbol error rate calculating...")
with open(Path(decode_dir,'best-hyp'),'r') as hyp_ref_in, \
     open(Path(decode_dir,'hyp.trn'),'w') as hyp_out, \
     open(Path(decode_dir,'ref.trn'),'w') as ref_out:
    for i,line in enumerate(hyp_ref_in.readlines()):
        foo = line.rstrip().split('\t')
        if len(foo) == 1:
            print(f"{' '.join(to_list(foo[0]))} ({i//1000}k_{i})", file=ref_out)
            print(f"({i//1000}k_{i})", file=hyp_out)
        elif len(foo) == 2:
            ref = foo[0]
            hyp = foo[1]
            print(f"{' '.join(to_list(ref))} ({i//1000}k_{i})", file=ref_out)
            print(f"{' '.join(to_list(hyp))} ({i//1000}k_{i})", file=hyp_out)
        else:
            raise ValueError("at most only ref and hyp")
res = run_cmd(['sclite','-r',Path(decode_dir,'ref.trn'),'trn', '-h', Path(decode_dir,'hyp.trn'),'trn','-i','rm','-o','all','stdout'])
logger.log(f"Write result to {Path(decode_dir,'result.txt')}",prefix='info')
with open(Path(decode_dir,'result.txt'),'w') as fout:
    print(res, file=fout)
er_rate = run_cmd(['grep','-e','Avg','-e','SPKR','-m','2',Path(decode_dir,'result.txt')])
print(er_rate)
ser = run_cmd(['grep','-e','Sum/Avg','-m','1',Path(decode_dir,'result.txt')])
ser = re.sub(' +',' ',ser).split(' ')[10]
comet_exp.log_other(f"ser({paras.decode_mode})",ser)
logger.log(f"Symbol ER: {ser}", prefix='test')
with open(Path(decode_dir,'ser'),'w') as fout:
    print(str(ser), file=fout)
################################################################################

### Cal WER ####################################################################
logger.notice("WER calculating...")
with open(Path(decode_dir,'best-hyp'),'r') as hyp_ref_in, \
     open(Path(decode_dir,'hyp-word.trn'),'w') as hyp_out, \
     open(Path(decode_dir,'ref-word.trn'),'w') as ref_out:
    for i,line in enumerate(hyp_ref_in.readlines()):
        foo = line.rstrip().split('\t')
        if len(foo) == 1:
            ref = spm.DecodePieces(to_list(foo[0]))
            print(f"{ref} ({i//1000}k_{i})", file=ref_out)
            print(f"({i//1000}k_{i})", file=hyp_out)
        elif len(foo) == 2:
            ref = spm.DecodePieces(to_list(foo[0]))
            hyp = spm.DecodePieces(to_list(foo[1]))
            print(f"{ref} ({i//1000}k_{i})", file=ref_out)
            print(f"{hyp} ({i//1000}k_{i})", file=hyp_out)
        else:
            raise ValueError("at most only ref and hyp")
res = run_cmd(['sclite','-r',Path(decode_dir,'ref-word.trn'),'trn', '-h', Path(decode_dir,'hyp-word.trn'),'trn','-i','rm','-o','all','stdout'])
logger.log(f"Write result to {Path(decode_dir,'result.wrd.txt')}",prefix='info')
with open(Path(decode_dir,'result.wrd.txt'),'w') as fout:
    print(res, file=fout)
er_rate = run_cmd(['grep','-e','Avg','-e','SPKR','-m','2',Path(decode_dir,'result.wrd.txt')])
print(er_rate)
wer = run_cmd(['grep','-e','Sum/Avg','-m','1',Path(decode_dir,'result.wrd.txt')])
wer = re.sub(' +',' ',wer).split(' ')[10]
logger.log(f"WER: {wer}", prefix='test')
comet_exp.log_other(f"wer({paras.decode_mode})",wer)
with open(Path(decode_dir,'wer'),'w') as fout:
    print(str(wer), file=fout)
#################################################################################

comet_exp.log_other('status','completed')

wc = run_cmd(['wc','-l',Path(decode_dir,'best-hyp')])
wc = wc.split(' ')[0]
comet_exp.log_other(f"#decode",wc)
