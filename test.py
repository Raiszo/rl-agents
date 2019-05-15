import gym
from rl_agents.ppo.policy import Actor


env = gym.make('Pendulum-v0')
is_continuous = isinstance(env.action_space, gym.spaces.Box)

actor = Actor(env.observation_space.shape, env.action_space.shape, is_continuous)

obs = env.reset()
x = actor.action(obs[None])
print(x)

# import unittest

# from rl_agents.ppo.policy import Actor

# actor = Actor()
# class TestActor(unittest.testCase):
    
#     def test_sample(self):
#         pass


# if if __name__ == '__main__':
#     unittest.main()
