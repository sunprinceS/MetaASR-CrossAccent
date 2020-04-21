import os
from yaml import dump
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from src.marcos import *
import src.monitor.logger as logger

class Dashboard:
    def __init__(self, config, paras, log_dir, train_type, resume=False):
        self.log_dir = log_dir
        self.global_step = 1
        self.tb_dir = self.log_dir.parents[7].joinpath('tensorboard').joinpath(*self.log_dir.parts[5:])

        #TODO: don't know the resume mechanism in tensorboard
        self.exp = SummaryWriter(self.tb_dir)

    def log_config(self, config):

        with open(self.log_dir.joinpath('config.yaml'),'w') as fout:
            yaml.dump(config, fout)


    def set_status(self, status):
        pass

    def step(self, n=1):
        self.global_step += n

    def set_step(self, global_step=1):
        self.global_step = global_step

    def log_info(self, prefix, info):
        for k, v in info.items():
            self.exp.add_scalar('/'.join([prefix, k]),float(v), global_step=self.global_step)

    def log_other(self, name, value):
        self.exp.add_scalar(name, value, global_step = self.global_step)

    def log_step(self):
        pass

    def add_figure(self, fig_name, data):
        self.exp.add_figure(fig_name, data, global_step = self.global_step)

    def check(self):
        pass
