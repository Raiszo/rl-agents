import tensorflow as tf
from math import ceil
import numpy as np

class VPG_Agent:
    def __init__(self, actor, critic, is_continuous, act_dim,
                 learning_rate=3e-4, ac_lr=3e-4, cr_lr=1e-3):
        self.actor = actor
        self.critic = critic

        self.is_continuous = is_continuous
        self.act_dim = act_dim

        self.actor_opt = tf.keras.optimizers.Adam(ac_lr)
        self.critic_opt = tf.keras.optimizers.Adam(cr_lr)
        self.opt = tf.keras.optimizers.Adam(ac_lr)
        # self.MSE = tf.keras.losses.MeanSquaredError()


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
            # Maybe should add arg trainning=True
            pi, logp_pi, dist, locs = self.actor(obs_no)
            logp_ac_n = dist.log_prob(ac_na)

            pg_loss = tf.reduce_mean(logp_ac_n * adv_n)
            if self.is_continuous:
                ent_loss = tf.reduce_mean(dist.entropy())
            else:
                ent_loss = 0.0


            loss = - pg_loss - 0.01*ent_loss

        tvars = self.actor.trainable_variables
        grad = tape.gradient(loss, tvars)
        self.actor_opt.apply_gradients(zip(grad, tvars)) 


    @tf.function
    def critic_step(self, obs_no, tval_n):
        with tf.GradientTape() as tape:
            pval_n = self.critic(obs_no)

            loss = tf.reduce_mean((pval_n - tval_n)**2)
            loss = 0.5 * loss
            # value_loss = self.MSE(
            #     y_true=tval_n,
            #     y_pred=pval_n,
            # )

        tvars = self.critic.trainable_variables
        grad = tape.gradient(loss, tvars)
        self.critic_opt.apply_gradients(zip(grad, tvars)) 
   

    def run_ite(self, obs_no, ac_na, log_prob_na, t_val_n, adv_n,
                epochs_actor, epochs_critic, batch_size=64):
        size = len(obs_no)
        train_indicies = np.arange(size)

        if len(ac_na.shape) == 1:
            acs = np.zeros((size, self.act_dim))
            acs[np.arange(size), ac_na] = 1
            ac_na = acs

        act_ds = tf.data.Dataset.from_tensor_slices((obs_no, ac_na, log_prob_na, adv_n))
        act_ds = act_ds.shuffle(512).batch(batch_size).repeat(epochs_actor)

        crt_ds = tf.data.Dataset.from_tensor_slices((obs_no, t_val_n))
        crt_ds = crt_ds.shuffle(512).batch(batch_size).repeat(epochs_critic)

        for obs, ac, logp, adv in act_ds:
            self.actor_step(obs, ac, adv)

        for obs, t_val in crt_ds:
            self.critic_step(obs, t_val)
