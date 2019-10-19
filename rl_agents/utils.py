import gym

def check_if_continuous(env) -> bool:
    return isinstance(env.action_space, gym.spaces.Box)
