import tensorflow as tf
from math import ceil
import numpy as np

class VPG_Agent:
    def __init__(self, actor, critic, is_continuous, act_dim,
                 learning_rate=3e-4):
        self.actor = actor
        self.critic = critic

        self.act_dim = act_dim

        self.actor_opt = tf.keras.optimizers.Adam(3e-4)
        self.critic_opt = tf.keras.optimizers.Adam(1e-3)

        self.MSE = tf.keras.losses.MeanSquaredError()

        # self.reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float64)

    @tf.function
    def act_stochastic(self, obs):
        pi, logp_pi, dist, loc = self.actor(obs[None])
        value = self.critic(obs[None])

        return pi[0], logp_pi[0], value[0]


    def act_deterministic(self, obs):
        _, _, _, loc = self.actor(obs[None])

        return loc[0]


    @tf.function
    def train_step_actor(self, obs_no, ac_na, adv_n):
        with tf.GradientTape() as act_tape:
            # Maybe should add arg trainning=True
            pi, logp_pi, dist, locs = self.actor(obs_no)
            logp_ac_n = dist.log_prob(ac_na)
            # entropy = dist.entropy()


            pg_loss = tf.reduce_mean(logp_ac_n * adv_n)
            # ent_loss = tf.reduce_mean(entropy)

            act_loss = - pg_loss

        act_grad = act_tape.gradient(act_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(act_grad, self.actor.trainable_variables))

    @tf.function
    def train_step_critic(self, obs_no, tval_n):
        with tf.GradientTape() as crt_tape:
            pval_n = self.critic(obs_no)

            value_loss = tf.reduce_mean((pval_n - tval_n)**2)
            # value_loss = self.MSE(
            #     y_true=tval_n,
            #     y_pred=pval_n,
            # )

            crt_loss = value_loss

        crt_grad = crt_tape.gradient(crt_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(crt_grad, self.critic.trainable_variables)) 
   

    @tf.function
    def train_step(self, obs_no, ac_na, adv_n, tval_n, old_logp_pi):

        with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
            # Maybe should add arg trainning=True
            pi, logp_pi, dist, locs = self.actor(obs_no)
            logp_ac_n = dist.log_prob(ac_na)
            # entropy = dist.entropy()
            pval_n = self.critic(obs_no)


            pg_loss = tf.reduce_mean(logp_ac_n * adv_n)
            # ent_loss = tf.reduce_mean(entropy)
            value_loss = self.MSE(
                y_true=tval_n,
                y_pred=pval_n,
            )

            act_loss = - pg_loss
            crt_loss = value_loss

        act_grad = act_tape.gradient(act_loss, self.actor.trainable_variables)
        crt_grad = crt_tape.gradient(crt_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(act_grad, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(crt_grad, self.critic.trainable_variables))


    def run_ite(self, obs_no, ac_na, log_prob_na, t_val_n, adv_n,
                batch_size=64):
        size = len(obs_no)
        train_indicies = np.arange(size)

        # print(log_prob_na)
        # return
        # A discrete env, so ac_na is shape (n), need (n,act_dim)
        if len(ac_na.shape) == 1:
            acs = np.zeros((size, self.act_dim))
            acs[np.arange(size), ac_na] = 1
            ac_na = acs
            # ac_na = tf.one_hot(ac_na, depth=self.act_dim)
            # lp = np.zeros((size, self.act_dim))
            # lp[np.arange(size), log_prob_na] = 1
            # log_prob_na = lp

        # print(ac_na)

        for i in range(int(ceil(size/batch_size))):
            start_idx = (i*batch_size)%size
            idx = train_indicies[start_idx:start_idx+batch_size]

            obs_no_b = obs_no[idx, :]
            ac_na_b = ac_na[idx, :]
            log_prob_na_b = log_prob_na[idx]

            self.train_step_actor(obs_no_b, ac_na_b, adv_n[idx])
            for _ in range(80):
                self.train_step_critic(obs_no_b, t_val_n[idx])
            # self.train_step(obs_no_b, ac_na_b, adv_n[idx], t_val_n[idx], log_prob_na_b)
