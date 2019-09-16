import tensorflow as tf
import numpy as np
import datetime
from os import path


from rl_agents.env_utils import rollouts_generator, get_adv_vtarg

class Sensei:
    def __init__(self, agent, alg_name, env_fn,
                 ite=200, horizon=2048,
                 epochs_actor=20, epochs_critic=20,
                 gamma=0.99, gae_lambda=0.95,
                 log_dir='logs'):
        self.agent = agent
        self.alg_name = alg_name
        self.env_fn = env_fn

        self.num_ite = ite
        self.horizon = horizon
        self.epochs_actor = epochs_actor
        self.epochs_critic = epochs_critic
        self.gamma = 0.99
        self.gae_lambda = gae_lambda

        self.log_dir = log_dir

        is_continuous = self.agent.is_continuous
        self.generator = rollouts_generator(agent, env_fn(), is_continuous, horizon)

        env = self.env_fn()
        folder_name = '{}_{}'.format(env.unwrapped.spec.id, self.alg_name)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = path.join(self.log_dir, folder_name, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        
    def train(self, batch_size=64) -> None:
        env = self.env_fn()

        for i in range(self.num_ite):
            rollout = self.generator.__next__()
            adv, target_value = get_adv_vtarg(rollout, lam=self.gae_lambda, gamma=self.gamma)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
            self.agent.run_ite(rollout['ob'], rollout['ac'], rollout['log_probs'], target_value, adv, epochs_actor=self.epochs_actor, epochs_critic=self.epochs_critic, batch_size=batch_size)
            with self.summary_writer.as_default():
                tf.summary.scalar('reward mean', np.array(rollout["ep_rets"]).mean(), step=i)

            log_dir = self.log_dir
            # log_dir_fn = lambda log_dir, name, i: path.join(log_dir, )
            if i % 50 == 0 or i == self.num_ite-1:
                self.agent.actor.save_weights(log_dir+'/_actor_'+str(i), save_format='tf')
                self.agent.critic.save_weights(log_dir+'/_critic_'+str(i), save_format='tf')


    def test(self):
        pass
