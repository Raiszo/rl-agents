import tensorflow as tf
from math import ceil
import numpy as np

from rl_agents.policies.gaussian import GaussianActor

class PPO_Agent:
    def __init__(self, actor, critic, act_dim,
                 epsilon=0.2, ac_lr=3e-4, cr_lr=1e-3):
        self.name = 'PPO'
        self.actor = actor
        self.critic = critic
        self.use_entropy = isinstance(actor, GaussianActor)

        self.act_dim = act_dim
        self.epsilon = epsilon

        self.actor_opt = tf.keras.optimizers.Adam(ac_lr)
        self.critic_opt = tf.keras.optimizers.Adam(cr_lr)

        # self.reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float64)

    @tf.function
    def act_stochastic(self, obs):
        pi, logp_pi, dist, loc = self.actor(obs[None])
        value = self.critic(obs[None])

        return pi[0], logp_pi[0], value[0]


    @tf.function
    def act_deterministic(self, obs):
        _, _, _, loc = self.actor(obs[None])
        # value = self.critic(obs[None])

        return loc[0]
        

    def surrogate_loss(self, new_logp, old_logp, adv):
        diff = new_logp - old_logp
        ratio = tf.exp(diff)

        clipped_ratio = tf.clip_by_value(ratio,
                                         1.0 - self.epsilon,
                                         1.0 + self.epsilon)
        surrogate_min = tf.minimum(ratio*adv, clipped_ratio*adv)

        return tf.reduce_mean(surrogate_min)


    @tf.function
    def actor_step(self, obs_no, ac_na, adv_n, old_logp_n):
        with tf.GradientTape() as tape:
            # Compute new log probs
            _, _, dist, _ = self.actor(obs_no)

            logp = dist.log_prob(ac_na)
            if len(logp.shape) > 1:
                new_logp_n = tf.reduce_sum(logp, axis=1)
            else:
                new_logp_n = logp


            surr_loss = self.surrogate_loss(new_logp_n, old_logp_n, adv_n)
            ent_loss = tf.reduce_mean(dist.entropy()) if self.use_entropy else 0.0

            loss = surr_loss - 0.01*ent_loss

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(gradients, self.actor.trainable_variables))


    @tf.function
    def critic_step(self, obs_no, tval_n):
        with tf.GradientTape() as tape:
            pval_n = self.critic(obs_no)
            pval_n = tf.squeeze(pval_n)

            loss = tf.reduce_mean((tval_n - pval_n)**2)
            loss = 0.5 * loss

        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(gradients, self.critic.trainable_variables))
    

    def run_ite(self, obs_no, ac_na, logp_n, t_val_n, adv_n,
                epochs_actor, epochs_critic, batch_size):
        """
        Remember the old school way to do it: 
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]
                obs_no[idx, :]
        
        Train actor with inptus: obs, ac, logp, adv
        """
        act_ds = tf.data.Dataset.from_tensor_slices((obs_no, ac_na, adv_n, logp_n))
        act_ds = act_ds.shuffle(512).batch(batch_size).repeat(epochs_critic)

        crt_ds = tf.data.Dataset.from_tensor_slices((obs_no, t_val_n))
        crt_ds = crt_ds.shuffle(512).batch(batch_size).repeat(epochs_actor)

        for obs, ac, adv, logp in act_ds:
            self.actor_step(obs, ac, adv, logp)

        for obs, t_val in crt_ds:
            self.critic_step(obs, t_val)
            

        # for epoch in range(epochs):
        #     for i in range(int(ceil(size/batch_size))):
        #         start_idx = (i*batch_size)%size
        #         idx = train_indicies[start_idx:start_idx+batch_size]
        #         # print(idx)

        #         obs_no_b = obs_no[idx, :]
        #         ac_na_b = ac_na[idx, :]
        #         log_prob_na_b = log_prob_na[idx]

        #         self.train_step(obs_no_b, ac_na_b, log_prob_na_b, adv_n[idx], t_val_n[idx])

