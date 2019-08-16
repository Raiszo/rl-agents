import tensorflow as tf
from math import ceil
import numpy as np

class PPO_Agent:
    def __init__(self,
                 actor,
                 critic,
                 epsilon=0.2, learning_rate=1e-4):
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
        loc, _, sample, log_prob = self.actor(obs[None], training=True)
        value = self.critic(obs[None])

        return loc[0], sample[0], log_prob[0], value[0]


    def act_deterministic(self, obs):
        action, _, _, _ = self.actor(obs[None], training=False)

        return action
        

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
        value_loss = tf.keras.losses.mean_squared_error(y_true=t_value_n, y_pred=p_value_n)
        value_loss = tf.reduce_mean(value_loss)

        return 0.5 * value_loss

    @tf.function
    def train_step(self, obs_no, ac_na, old_log_prob_n, adv_n,
                   true_value_n):

        # with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
        #     _, _, _, dist = self.actor(obs_no, training=True)
        #     new_log_prob_n = dist.log_prob(ac_na)
        #     entropies = dist.entropy()

        #     # Need to recompute this to record the gradient in the gradient tape
        #     pred_value_n = self.critic(obs_no)

        #     # print(new_log_probs)
        #     # print(entropies)
        #     act_loss = self.actor_loss(new_log_prob_n, old_log_prob_n, entropies, adv_n)
        #     crt_loss = self.critic_loss(true_value_n, pred_value_n)

        # gradients_of_actor = act_tape.gradient(act_loss, self.actor.trainable_variables)
        # gradients_of_critic = crt_tape.gradient(crt_loss, self.critic.trainable_variables)

        # self.actor_opt.apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        # self.critic_opt.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))

        # tf.print('std', self.actor.sample.std)
        with tf.GradientTape() as tape:
            locs, dist, _, _ = self.actor(obs_no, training=True)
            # tf.print('obss', obs_no[0:5])
            # tf.print('locs', locs[0:5])
            # tf.print('acs', ac_na[0:5])
            # tf.print('lp', old_log_prob_n[0:5])
            new_log_prob_n = dist.log_prob(ac_na)
            entropies = dist.entropy()

            # Need to recompute this to record the gradient in the gradient tape
            pred_value_n = self.critic(obs_no)

            # tf.print('new', new_log_prob_n[0:5])
            # tf.print('old', old_log_prob_n[0:5])
            # print(entropies)
            act_loss = self.actor_loss(new_log_prob_n, old_log_prob_n, entropies, adv_n)
            crt_loss = self.critic_loss(true_value_n, pred_value_n)

            loss = act_loss + crt_loss

        variables = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(gradients, variables))


    @tf.function
    def my_step(self, obs_no, ac_na, old_log_prob_n):
        locs, dist, _, _ = self.actor(obs_no, training=True)

        # tf.print('obss', obs_no[0:5])
        # tf.print('locs', locs[0:5])
        # tf.print('acs', ac_na[0:5])
        # tf.print('lp', old_log_prob_n[0:5])
        new_log_prob_n = dist.log_prob(ac_na)
        entropies = dist.entropy()
        
        diff = new_log_prob_n - old_log_prob_n
        ratio = tf.exp(diff)
        # tf.print('ratio shape', ratio.shape)
        # tf.print('diff:', tf.reduce_mean(diff))

    def run_ite(self, obs_no, ac_na, log_prob_na, locs_na, t_val_n, adv_n,
                epochs, batch_size=64):
        size = len(obs_no)
        train_indicies = np.arange(size)

        for epoch in range(epochs):
            # for i in range(3):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]
                # print(idx)

                obs_no_b = obs_no[idx, :]
                locs_na_b = locs_na[idx, :]
                ac_na_b = ac_na[idx, :]
                log_prob_na_b = log_prob_na[idx, :]

                # print(epoch, i)
                # print('------ batch ------')
                # print('observations', obs_no_b[0:5])
                # print('locs', locs_na_b[0:5])
                # print('actions', ac_na_b[0:5])
                # print('log_probs', log_prob_na_b[0:5])
                # print(obs_no_b.shape)
                # print(ac_na_b.shape)
                # print(log_prob_na_b.shape)
                # print('------ batch ------')

                # self.my_step(obs_no_b, ac_na_b, log_prob_na_b)
                self.train_step(obs_no_b, ac_na_b, log_prob_na_b, adv_n[idx], t_val_n[idx])
                # self.train_step(obs[idx, :], ac[idx, :], log_prob[idx], adv[idx], t_val[idx])
