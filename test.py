import gym
import numpy as np
from pathlib import Path
# from rl_agents.ppo.policy import Actor, Critic
# from rl_agents.ppo.agent import PPO_Agent
from rl_agents.vpg.agent import VPG_Agent
from rl_agents.models import GaussianActor, CategoricalActor, Critic

def render(agent, env, recorder=None):
    obs = env.reset()
    done = False

    total_rew = 0
    while not done:
        frame = env.render(mode='rgb_array')
        if recorder: recorder.write(frame)
        ac = agent.act_deterministic(obs)

        ob, rew, done, _ = env.step(ac.numpy())
        # print(rew)
        total_rew += np.sum(rew)

    print('Total reward at testig', total_rew)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Render agent actions on environment')
    parser.add_argument('--logs_dir', help='logs directory')
    parser.add_argument('--env', help='environment name')
    parser.add_argument('--ite', help='iteration of experiment')
    args = parser.parse_args()


    weights_actor_file = args.logs_dir+'_actor_'+args.ite
    weights_critic_file = args.logs_dir+'_critic_'+args.ite
    # assert Path(weights_actor_file).is_file(), 'Actor wights file does not exist >:v' 
        

    env = gym.make(args.env)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape if is_continuous else env.action_space.n


    actor = GaussianActor(obs_dim, act_dim) if is_continuous else CategoricalActor(obs_dim, act_dim)
    critic = Critic(obs_dim)

    rand_obs = env.observation_space.sample().astype(np.float64)
    a = actor(rand_obs[None])
    b = critic(rand_obs[None])

    # print(a, b)

    actor.load_weights(weights_actor_file)
    critic.load_weights(weights_critic_file)
    jen = VPG_Agent(actor, critic, is_continuous, act_dim)


    render(jen, env)
    
    
if __name__ == '__main__':
    main()
