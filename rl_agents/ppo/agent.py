import tensorflow as tf
from math import ceil
import numpy as np

class PPO_Agent:
    def __init__(self, actor, critic, is_continuous, act_dim,
                 epsilon=0.2, ac_lr=3e-4, cr_lr=1e-3):
        self.actor = actor
        self.critic = critic

        self.is_continuous = is_continuous
        self.act_dim = act_dim
        self.epsilon = epsilon

        self.actor_opt = tf.keras.optimizers.Adam(ac_lr)
        self.critic_opt = tf.keras.optimizers.Adam(cr_lr)
        self.opt = tf.keras.optimizers.Adam(ac_lr)

        # self.reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float64)

    @tf.function
    def act_stochastic(self, obs):
        pi, logp_pi, dist, loc = self.actor(obs[None])
        value = self.critic(obs[None])

        return pi[0], logp_pi[0], value[0]


    def act_deterministic(self, obs):
        _, _, _, loc = self.actor(obs[None])

        return loc[0]
        

    def surrogate_loss(self, new_logp_pi, old_logp_pi, advs):
        diff = new_logp_pi - old_logp_pi
        ratio = tf.exp(diff)

        clipped_ratio = tf.clip_by_value(ratio,
                                         1.0 - self.epsilon,
                                         1.0 + self.epsilon)
        surrogate_min = tf.minimum(ratio*advs, clipped_ratio*advs)

        return - tf.reduce_mean(surrogate_min)


    @tf.function
    def actor_step(self, obs_no, ac_na, old_log_prob_na, adv_n):
        with tf.GradientTape() as tape:
            # Compute new log probs
            _, _, dist, _ = self.actor(obs_no)
            new_log_prob_na = dist.log_prob(ac_na)

            # Compute surrogate loss
            surr_loss = self.surrogate_loss(new_log_prob_na, old_log_prob_na, adv_n)
            # Can only calculate a valid entropy for gaussian distribution
            if self.is_continuous:
                ent_loss = tf.reduce_mean(dist.entropy())
            else:
                ent_loss = 0.0

            loss = surr_loss - 0.01*ent_loss

        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(gradients, self.actor.trainable_variables))


    @tf.function
    def critic_step(self, obs_no, true_value_n):
        with tf.GradientTape() as tape:
            pred_value_n = self.critic(obs_no)
            loss = tf.reduce_mean((pred_value_n - true_value_n)**2)
            loss = 0.5 * loss

        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(gradients, self.critic.trainable_variables))
    

    def run_ite(self, obs_no, ac_na, log_prob_na, t_val_n, adv_n,
                epochs, batch_size):
        size = len(obs_no)
        train_indicies = np.arange(size)

        if len(ac_na.shape) == 1:
            acs = np.zeros((size, self.act_dim))
            acs[np.arange(size), ac_na] = 1
            ac_na = acs

        """
        Remember the old school way to do it: 
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]
                obs_no[idx, :]
        
        Train actor with inptus: obs, ac, logp, adv
        """
        act_ds = tf.data.Dataset.from_tensor_slices((obs_no, ac_na, log_prob_na, adv_n))
        act_ds = act_ds.shuffle(512).batch(batch_size).repeat(epochs)

        crt_ds = tf.data.Dataset.from_tensor_slices((obs_no, t_val_n))
        crt_ds = crt_ds.shuffle(512).batch(batch_size).repeat(epochs)

        for obs, ac, logp, adv in act_ds:
            self.actor_step(obs, ac, logp, adv)

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

