import numpy as np
import gym

def is_continuous(env):
    return isinstance(env.action_space, gym.spaces.Box)
