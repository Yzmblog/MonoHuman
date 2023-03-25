import os
import argparse

import torch

from third_parties.yacs import CfgNode as CN

# pylint: disable=redefined-outer-name

_C = CN()

# "resume" should be train options but we lift it up for cmd line convenience
_C.resume = False

# current iteration -- set a very large value for evaluation
_C.eval_iter = 10000000

# for rendering
_C.render_folder_name = ""
_C.ignore_non_rigid_motions = False
_C.render_skip = 1
_C.render_frames = 100

# for data loader
_C.num_workers = 16


def get_cfg_defaults():
    return _C.clone()


def parse_cfg(cfg):
    cfg.logdir = os.path.join('experiments', cfg.category, cfg.task, cfg.subject, cfg.experiment)


def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg)
        
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True, type=str)
parser.add_argument("--type", default="skip", type=str)
parser.add_argument("--test_view", required=False, type=str)
parser.add_argument("--gt_path", required=False, type=str)
parser.add_argument("--pre_path", required=False, type=str)
parser.add_argument("--masks_path", required=False, type=str)
parser.add_argument("--iter", required=False, type=str)
parser.add_argument("--id", required=False, type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)
