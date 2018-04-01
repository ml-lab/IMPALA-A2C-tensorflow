from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import go_vncdriver
import os
import time
import sys, signal
import tensorflow as tf

import trainer
import utils as ut
from envs import create_env

logger = ut.logging.get_logger()


def main(_):
    from config import get_args
    args = get_args()

    assert args.num_gpu > 0, \
            "You need at least one GPU to train a SPIRAL agent"

    ut.train.set_global_seed(args.seed)
    spec = ut.tf.cluster_spec(
            args.num_workers, 1, args.start_port)
    cluster = tf.train.ClusterSpec(spec)
    cluster_def = cluster.as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    #############################
    # Prepare common envs
    #############################

    env = create_env(
            args.env,
            client_id=str(args.task),
            remotes=args.remotes)


    queue_shapes = [
            ['actions', [1]],
            ['states', list(env.observation_space.shape)],
            ['rewards', []],
            ['values', [1]],
            ['features', [2, args.lstm_size]],
    ]

    for idx, (name, shape) in enumerate(queue_shapes):
        length = args.num_local_steps
        queue_shapes[idx][1] = [length] + shape

    queue_shapes.extend([
            ('r', []),
    ])

    trajectory_queue_size = \
            args.policy_batch_size * min(5, args.num_workers)

    #############################
    # Run
    #############################

    if args.task == 0:
        ut.train.save_args(args)

    if args.job_name == "worker":
        tf_config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2)

        server = tf.train.Server(
                cluster_def,
                job_name="worker",
                task_index=args.task,
                config=tf_config)
        trainer.train(args, server, cluster, env, queue_shapes,
                      trajectory_queue_size)
    else:
        del env

        server = tf.train.Server(
                cluster_def, job_name="ps", task_index=args.task,
                config=tf.ConfigProto(device_filters=["/job:ps"]))

        with tf.device(f'/job:ps/task:{args.task}'):
            queue_size = args.policy_batch_size * args.num_workers

            queue = tf.FIFOQueue(
                    trajectory_queue_size,
                    [tf.float32] * len(queue_shapes),
                    shapes=[shape for _, shape in queue_shapes],
                    shared_name='queue')

        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
