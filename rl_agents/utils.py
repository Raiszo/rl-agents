from gym.spaces import Box, Discrete

from rl_agents.policies.categorical import CategoricalActor
from rl_agents.policies.gaussian import GaussianActor
from rl_agents.common import Critic

def check_if_continuous(env) -> bool:
    return isinstance(env.action_space, gym.spaces.Box)

def get_actor_critic(env):
    obs_dim = env.observation_space.shape
    is_continuous = check_if_continuous(env)

    if isinstance(env.action_space, Box):
        act_dim = a.shape.as_list()[-1]
        actor = GaussianActor(obs_dim, act_dim)
    elif isinstance(env.action_space, Discrete):
        act_dim = action_space.n
        actor = CategoricalActor(obs_dim, act_dim)
    else:
        raise Exception('Invalid Env')

    critic = Critic(obs_dim)

    return actor, critic
