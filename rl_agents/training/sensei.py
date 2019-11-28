import tensorflow as tf
import numpy as np
import datetime
from os import path
from gym.spaces import Box, Discrete

class ExperimentRunner:
    def __init__(self, agent, env, buff):
        self.agent = agent
        self.env = env
        self.continuous = isinstance(env.action_space, Box)
        self.buff = buff
        self.num_ite = 0

        self.ep_rets = []
        self.ite = 0

        self.last_obs = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.ite == 0:
            obs = self.env.reset()
            self.cur_ep_ret = 0
        elif self.ite > 0 and self.ite < self.num_ite:
            obs = self.last_obs
        else:
            self.ite = 0
            raise StopIteration

        ep_rets = []
        for i in range(len(self.buff)):
            act, log_prob, val = self.agent.act_stochastic(obs)
            action = act.numpy() if self.continuous else np.argmax(act.numpy())
            # print(log_prob)
            obs, rew, new, _ = self.env.step(action)
            print(rew)
            self.cur_ep_ret += rew

            self.buff.store(obs, act, rew, val, log_prob)

            if new:
                print('new')
                _, _, last_val = self.agent.act_stochastic(obs)
                self.buff.finish_path(last_val.numpy()[0])

                ep_rets.append(self.cur_ep_ret)

                # print('newww', i)
                obs = self.env.reset()
                self.cur_ep_ret = 0

        # finish path if the episode is not over yet
        if not new:
            self.buff.finish_path()

        self.ite += 1
        self.last_obs = obs
        print(ep_rets)
        ep_rets = np.array(ep_rets)

        return self.buff.get(ep_rets)

class Sensei:
    def __init__(self, agent, alg_name, env_fn, buff,
                 epochs_actor=20, epochs_critic=20,
                 gamma=0.99, gae_lambda=0.95,
                 log_dir='logs'):
        self.agent = agent
        self.alg_name = alg_name
        self.env_fn = env_fn
        env = env_fn()
        self.experiment_runner = ExperimentRunner(agent, env, buff)

        self.epochs_actor = epochs_actor
        self.epochs_critic = epochs_critic
        self.gamma = 0.99
        self.gae_lambda = gae_lambda

        self.log_dir = log_dir

        folder_name = '{}_{}'.format(env.unwrapped.spec.id, self.alg_name)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = path.join(self.log_dir, folder_name, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    
    def train(self, num_ite, agent_kwargs, record=True, batch_size=64):
        self.agent.setup_training(**agent_kwargs)
        # Set experiment_runner's iterations to run
        self.experiment_runner.num_ite = num_ite

        for i, rollout in enumerate(self.experiment_runner):
            self.agent.run_ite(rollout['obs'], rollout['act'], rollout['logp'],
                               rollout["ret"], rollout["adv"],
                               epochs_actor=self.epochs_actor, epochs_critic=self.epochs_critic,
                               batch_size=batch_size)

            if record:
                with self.summary_writer.as_default():
                    if len(rollout["ep_rets"]) > 0:
                        print(i, rollout["ep_rets"].mean())
                        tf.summary.scalar('reward mean', rollout["ep_rets"].mean(), step=i)

                log_dir = self.log_dir
                # log_dir_fn = lambda log_dir, name, i: path.join(log_dir, )
                if i % 50 == 0 or i == num_ite-1:
                    self.agent.actor.save_weights(log_dir+'/_actor_'+str(i), save_format='tf')
                    self.agent.critic.save_weights(log_dir+'/_critic_'+str(i), save_format='tf')
