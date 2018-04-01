from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from pathlib import Path

import utils as ut

logger = ut.logging.get_logger()


parser = argparse.ArgumentParser()

# model
model_arg = ut.args.add_argument_group(parser, 'model')
model_arg.add_argument('--lstm-size', default=256, type=int)


# environment
env_arg = ut.args.add_argument_group(parser, 'environment')
env_arg.add_argument('--env', default="PongDeterministic-v4")
env_arg.add_argument('--screen_size', type=int, default=64)
env_arg.add_argument('--color_channel', type=int, default=3, choices=[3, 1])


# train
train_arg = ut.args.add_argument_group(parser, 'train')
train_arg.add_argument('--policy_lr', default=1e-4, type=float)
train_arg.add_argument('--entropy_coeff', default=0.01, type=float)
train_arg.add_argument('--grad_clip', default=40, type=int)
train_arg.add_argument('--policy_batch_size', default=64, type=int)
train_arg.add_argument('--num_local_steps', default=20, type=int)


# distributed
dist_arg = ut.args.add_argument_group(parser, 'distributed')
dist_arg.add_argument('--task', default=0, type=int)
dist_arg.add_argument('--remotes', default=None, type=str)
dist_arg.add_argument('--job_name', default="worker")
dist_arg.add_argument('--num_workers', default=4, type=int)
dist_arg.add_argument('--start_port', default=13333, type=int)


# Misc
misc_arg = ut.args.add_argument_group(parser, 'misc')
misc_arg.add_argument('--num_gpu', type=int, default=1,
                      choices=[1, 2])
misc_arg.add_argument('--log_dir', type=Path, default='logs')
misc_arg.add_argument('--load_path', type=Path, default=None)
misc_arg.add_argument('--log_level', type=str, default='INFO',
                      choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--seed', type=int, default=123)
misc_arg.add_argument('--dry_run', action='store_true')
misc_arg.add_argument('--tb_port', type=int, default=12345)


def get_args(group_name=None, parse_unknown=False):
    if parse_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    ##############################
    # Preprocess or filter args
    ##############################
    assert args.num_workers > 1, \
            "num_workers should be larger than 1 (policy, worker)"

    if parse_unknown:
        return args, unknown
    return args
