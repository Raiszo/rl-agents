import tensorflow as tf
from math import ceil
import numpy as np

from rl_agents.policies.gaussian import GaussianActor

class VPG_Agent:
    def __init__(self, actor, critic,
                 actor_lr=3e-4, critic_lr=1e-3, logger=None):
        self.name = 'VPG'
        self.actor = actor
        self.critic = critic
        self.use_entropy = isinstance(actor, GaussianActor)

        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(critic_lr)

        self.logger = logger
        if self.logger:
            self.actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
            self.critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
            self.summary_writer = self.logger.summary_writer


    @tf.function
    def act_stochastic(self, obs):
        pi, logp_pi, dist, loc = self.actor(obs[None])
        value = self.critic(obs[None])

        return pi[0], logp_pi[0], value[0]


    @tf.function
    def act_deterministic(self, obs):
        _, _, _, loc = self.actor(obs[None])

        return loc[0]


    @tf.function
    def actor_step(self, obs_no, ac_na, adv_n):
        with tf.GradientTape() as tape:
            pi, logp_pi, dist, locs = self.actor(obs_no)

            # Log probs for action dim > 1 are the sum of the result from dist.log_prob
            logp = dist.log_prob(ac_na)
            if len(logp.shape) > 1:
                logp_ac_n = tf.reduce_sum(logp, axis=1)
            else:
                logp_ac_n = logp
                
            pg_loss = tf.reduce_mean(logp_ac_n * adv_n)
            # ent_loss = tf.reduce_mean(dist.entropy()) if self.use_entropy else 0.0

            # loss = - pg_loss - 0.01*ent_loss
            loss = - pg_loss

        grad = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grad, self.actor.trainable_variables))

        if self.logger:
            self.actor_loss(loss)

    @tf.function
    def critic_step(self, obs_no, tval_n):
        with tf.GradientTape() as tape:
            pval_n = self.critic(obs_no)
            pval_n = tf.squeeze(pval_n)

            loss = tf.reduce_mean((tval_n - pval_n)**2)
            loss = 0.5 * loss

        grad = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_variables))

        if self.logger:
            self.critic_loss(loss)

    # @tf.function
    def run_ite(self, obs_no, ac_na, logp_n, t_val_n, adv_n,
                epochs_actor, epochs_critic, batch_size, i):

        act_ds = tf.data.Dataset.from_tensor_slices((obs_no, ac_na, adv_n))
        act_ds = act_ds.shuffle(512).batch(batch_size).repeat(epochs_actor)

        crt_ds = tf.data.Dataset.from_tensor_slices((obs_no, t_val_n))
        crt_ds = crt_ds.shuffle(512).batch(batch_size).repeat(epochs_critic)

        for obs, ac, adv in act_ds:
            self.actor_step(obs, ac, adv)

        for obs, t_val in crt_ds:
            self.critic_step(obs, t_val)

        if self.logger:
            with self.summary_writer.as_default():
                tf.summary.scalar('actor_loss', self.actor_loss.result(), step=i)
                tf.summary.scalar('critic_loss', self.critic_loss.result(), step=i)

            self.actor_loss.reset_states()
            self.critic_loss.reset_states()
