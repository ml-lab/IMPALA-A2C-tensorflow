from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np
import scipy.signal
from collections import namedtuple

import utils as ut


logger = ut.logging.get_logger()

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "features", "c"])


def discount(x, gamma):
    return scipy.signal.lfilter(
            [1], [1, -gamma], x[:,::-1], axis=1)[:,::-1]

def flatten_first_two(x):
    return np.reshape(x, [-1] + list(x.shape)[2:])

def multiple_process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout['states'])
    batch_a = np.asarray(rollout['actions'])
    rewards = np.asarray(rollout['rewards'])
    vpred_t = np.hstack(
            [rollout['values'][:,:,0], np.expand_dims(rollout['r'], -1)])

    rewards_plus_v = np.hstack(
            [rollout['rewards'], np.expand_dims(rollout['r'], -1)])
    batch_r = discount(rewards_plus_v, gamma)[:,:-1]
    delta_t = rewards + gamma * vpred_t[:,1:] - vpred_t[:,:-1]

    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout['features'][:,0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, features)


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.features = []

    def add(self, state, action, reward, value, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.features += [features]

    def extend(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.features.extend(other.features)


class WorkerThread(threading.Thread):
    def __init__(self, env, policy,
                 traj_enqueues, traj_placeholders, traj_size):
        threading.Thread.__init__(self)

        self.env = env
        self.sess = None
        self.daemon = True
        self.policy = policy
        self.last_features = None
        self.summary_writer = None
        self.num_local_steps = env.episode_length

        self.traj_enqueues = traj_enqueues
        self.traj_placeholders = traj_placeholders
        self.traj_size = traj_size

    def start_thread(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(
                self.env, self.policy, self.num_local_steps,
                self.summary_writer)
        while True:
            out = next(rollout_provider)

            feed_dict = {
                    self.traj_placeholders['actions']: out.actions,
                    self.traj_placeholders['states']: out.states,
                    self.traj_placeholders['rewards']: out.rewards,
                    self.traj_placeholders['values']: out.values,
                    self.traj_placeholders['features']: out.features,
                    self.traj_placeholders['r']: out.r,
            }
            if self.env.conditional:
                feed_dict.update({
                        self.traj_placeholders['conditions']: out.conditions,
                })

            for k, v in feed_dict.items():
                if isinstance(v, list):
                    feed_dict[k] = np.array(v)

            fetches = [
                    self.traj_size,
                    self.traj_enqueues
            ]

            out = self.sess.run(fetches, feed_dict)
            logger.info(f"# traj: {out[0]}")


def env_runner(env, policy, num_local_steps, summary_writer):
    last_state = env.reset()
    last_features = policy.get_initial_features(1, flat=True)

    length = 0
    rewards = 0

    while True:
        rollout = PartialRollout()

        last_action = env.initial_action

        for _ in range(num_local_steps):
            fetched = policy.act(
                    last_state, last_action, *last_features, condition)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            action = [np.argmax(action[name]) for name in env.acs]
            state, reward, terminal, info = env.step(action)

            # collect the experience
            rollout.add(last_state, action, reward,
                        value_, last_features, condition)
            length += 1

            # TODO: discriminator communication to get reward
            rewards += reward

            last_state = state
            last_action = action
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

        last_state = env.reset()
        last_features = policy.get_initial_features(1, flat=True)
        logger.debug(
                f"Episode finished. Sum of rewards: {rewards:.5f}." \
                f"Length: {length}.")

        length = 0
        rewards = 0

        rollout.states += [state]

        # once we have enough experience, yield it,
        # and have the ThreadRunner place it on a queue
        yield rollout
