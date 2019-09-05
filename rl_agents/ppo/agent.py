import tensorflow as tf
from math import ceil
import numpy as np

class PPO_Agent:
    def __init__(self, actor, critic, is_continuous, act_dim,
                 epsilon=0.2, learning_rate=3e-4):
        self.actor = actor
        self.critic = critic

        self.epsilon = epsilon
        self.alfa = learning_rate
        # self.actor_opt = tf.keras.optimizers.Adam(learning_rate)
        # self.critic_opt = tf.keras.optimizers.Adam(learning_rate)
        self.opt = tf.keras.optimizers.Adam(learning_rate)

        # self.reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float64)

    @tf.function
    def act_stochastic(self, obs):
        pi, logp_pi, dist, loc = self.actor(obs[None])
        value = self.critic(obs[None])

        return pi[0], logp_pi[0], value[0]


    def act_deterministic(self, obs):
        _, _, _, loc = self.actor(obs[None])

        return loc[0]
        

    def actor_loss(self, new_log_probs, old_log_probs, entropy, advs):
        diff = new_log_probs - old_log_probs
        ratio = tf.exp(diff)
        # tf.print('ratio shape', ratio.shape)
        # tf.print('diff:', tf.reduce_mean(diff))
        clipped_ratio = tf.clip_by_value(ratio,
                                         1.0 - self.epsilon,
                                         1.0 + self.epsilon)
        surrogate_min = tf.minimum(ratio*advs, clipped_ratio*advs)
        surrogate_loss = tf.reduce_mean(surrogate_min)
        entropy_loss = tf.reduce_mean(entropy)
        
        return - surrogate_loss - 0.01*entropy_loss


    def critic_loss(self, t_value_n, p_value_n):
        value_loss = tf.reduce_mean((p_value_n - t_value_n)**2)

        return 0.5 * value_loss

    @tf.function
    def train_step(self, obs_no, ac_na, old_log_prob_n, adv_n,
                   true_value_n):

        with tf.GradientTape() as tape:
            pi, logp_pi, dist, locs = self.actor(obs_no)
            new_log_prob_n = dist.log_prob(ac_na)
            entropies = dist.entropy()

            # Need to recompute this to record the gradient in the gradient tape
            pred_value_n = self.critic(obs_no)

            act_loss = self.actor_loss(new_log_prob_n, old_log_prob_n, entropies, adv_n)
            crt_loss = self.critic_loss(true_value_n, pred_value_n)

            loss = act_loss + crt_loss

        variables = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(gradients, variables))


    def run_ite(self, obs_no, ac_na, log_prob_na, t_val_n, adv_n,
                epochs, batch_size=64):
        size = len(obs_no)
        train_indicies = np.arange(size)

        if len(ac_na.shape) == 1:
            acs = np.zeros((size, self.act_dim))
            acs[np.arange(size), ac_na] = 1
            ac_na = acs

        for epoch in range(epochs):
            # for i in range(3):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]
                # print(idx)

                obs_no_b = obs_no[idx, :]
                ac_na_b = ac_na[idx, :]
                log_prob_na_b = log_prob_na[idx, :]

                self.train_step(obs_no_b, ac_na_b, log_prob_na_b, adv_n[idx], t_val_n[idx])

