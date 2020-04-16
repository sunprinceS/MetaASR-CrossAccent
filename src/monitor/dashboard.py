import os
from pathlib import Path
from comet_ml import Experiment, ExistingExperiment

from src.marcos import *
import src.monitor.logger as logger

class Dashboard:
    """Record training/evaluation statistics to comet
    :param Path log_dir
    :param list taskid_to_name
    """
    def __init__(self, config, paras, log_dir, train_type, resume=False):
        self.log_dir = log_dir
        self.expkey_f = Path(self.log_dir, 'exp_key')
        self.global_step = 1

        if resume:
            assert self.expkey_f.exists(), f"Cannot find comet exp key in {self.log_dir}"
            with open(Path(self.log_dir,'exp_key'),'r') as f:
                exp_key = f.read().strip()
            self.exp = ExistingExperiment(previous_experiment=exp_key,
                                                project_name=COMET_PROJECT_NAME, 
                                                workspace=COMET_WORKSPACE,
                                                auto_output_logging=None,
                                                auto_metric_logging=None,
                                                display_summary_level=0,
                                                )
        else:
            self.exp = Experiment(project_name=COMET_PROJECT_NAME,
                                  workspace=COMET_WORKSPACE,
                                  auto_output_logging=None,
                                  auto_metric_logging=None,
                                  display_summary_level=0,
                                  )
            #TODO: is there exists better way to do this?
            with open(self.expkey_f, 'w') as f:
                print(self.exp.get_key(),file=f)

            self.exp.log_other('seed', paras.seed)
            self.log_config(config)
            if train_type == 'evaluation':
                if paras.pretrain:
                    self.exp.set_name(f"{paras.pretrain_suffix}-{paras.eval_suffix}")
                    self.exp.add_tags([paras.pretrain_suffix, config['solver']['setting'], paras.accent, paras.algo, paras.eval_suffix])
                    if paras.pretrain_model_path:
                        self.exp.log_other("pretrain-model-path", paras.pretrain_model_path)
                    else:
                        self.exp.log_other("pretrain-runs", paras.pretrain_runs)
                        self.exp.log_other("pretrain-setting", paras.pretrain_setting)
                        self.exp.log_other("pretrain-tgt-accent", paras.pretrain_tgt_accent)
                else:
                    self.exp.set_name(paras.eval_suffix)
                    self.exp.add_tags(["mono", config['solver']['setting'], paras.accent])
            else:
                self.exp.set_name(paras.pretrain_suffix)
                self.exp.log_others({f"accent{i}": k for i,k in enumerate(paras.pretrain_accents)})
                self.exp.log_other('accent', paras.tgt_accent)
                self.exp.add_tags([paras.algo,config['solver']['setting'], paras.tgt_accent])
            #TODO: Need to add pretrain setting

        ##slurm-related
        hostname = os.uname()[1]
        if len(hostname.split('.')) == 2 and hostname.split('.')[1] == 'speech':
            logger.notice(f"Running on Battleship {hostname}")
            self.exp.log_other('jobid',int(os.getenv('SLURM_JOBID')))
        else:
            logger.notice(f"Running on {hostname}")


    def log_config(self,config):
        #NOTE: depth at most 2
        for block in config:
            for n, p in config[block].items():
                if isinstance(p, dict):
                    self.exp.log_parameters(p, prefix=f'{block}-{n}')
                else:
                    self.exp.log_parameter(f'{block}-{n}', p)

    def set_status(self,status):
        self.exp.log_other('status',status)

    def step(self, n=1):
        self.global_step += n

    def set_step(self, global_step=1):
        self.global_step = global_step

    def log_info(self, prefix, info):
        self.exp.log_metrics({k: float(v) for k, v in info.items()}, prefix=prefix, step=self.global_step)

    def log_step(self):
        self.exp.log_other('step',self.global_step)

    def add_figure(self, fig_name, data):
        self.exp.log_figure(figure_name=fig_name, figure=data, step=self.global_step)

    def check(self):
        if not self.exp.alive:
            logger.warning("Comet logging stopped")

    # def add_histogram(self, param, *tags):
        # self.writer.add_histogram('-'.join(tags), param, global_step = self.global_step)
