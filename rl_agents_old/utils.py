import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from os import path
import datetime

from rl_agents.policies.categorical import CategoricalActor
from rl_agents.policies.gaussian import GaussianActor
from rl_agents.common import Critic

def get_actor_critic(env):
    obs_dim = env.observation_space.shape
    env_as = env.action_space

    if isinstance(env_as, Box):
        act_out_dim = env_as.shape[-1]
        actor = GaussianActor(obs_dim, act_out_dim)
    elif isinstance(env_as, Discrete):
        act_out_dim = env_as.n
        actor = CategoricalActor(obs_dim, act_out_dim)
    else:
        raise Exception('Invalid Env')

    critic = Critic(obs_dim)

    return actor, critic


def simple_run(env, agent):
    continuous = isinstance(env.action_space, Box)

    obs = env.reset()
    done = False
    while not done:
        act, log_prob, val = agent.act_stochastic(obs)
        action = act.numpy() if continuous else np.argmax(act.numpy())

        obs, rew, done, _ = env.step(action)
        print('rew', rew)


class Logger:
    def __init__(self, log_dir, base_path='logs'):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._dir = path.join(base_path, log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(self._dir)

    def __call__(self):
        return self._dir
        
