import numpy as np
from gym.spaces import Box, Discrete

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
