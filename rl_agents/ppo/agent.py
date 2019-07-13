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
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate)

        self.reward_metric = tf.keras.metrics.Mean('reward', dtype=tf.float64)

    @tf.function
    def act_stochastic(self, obs):
        action, _, log_prob, _ = self.actor(obs[None], training=True)
        value = self.critic(obs[None])

        return action[0], value[0], log_prob[0]


    def get_distributions(self, obs):
        """
        Do not add a dim to obs, this will always have a batch
        """
        _, _, _, dist = self.actor(obs, training=True)

        return dist     


    def act_deterministic(self, obs):
        action, _, _, _ = self.actor.predict(obs)

        return action
        

    def actor_loss(self, new_log_probs, old_log_probs, entropy, advs):
        ratio = tf.exp(new_log_probs - old_log_probs)
        # tf.print('ratio:', tf.reduce_mean(ratio))
        clipped_ratio = tf.clip_by_value(ratio,
                                         1.0 - self.epsilon,
                                         1.0 + self.epsilon)
        surrogate_min = tf.minimum(ratio*advs, clipped_ratio*advs)
        surrogate_loss = tf.reduce_mean(surrogate_min)
        entropy_loss = tf.reduce_mean(entropy)
        
        return surrogate_loss + entropy_loss


    def critic_loss(self, t_value_n, p_value_n):
        value_loss = tf.keras.losses.mean_squared_error(y_true=t_value_n, y_pred=p_value_n)
        value_loss = tf.reduce_mean(value_loss)

        return value_loss


    @tf.function
    def train_step(self, obs_no, ac_na, log_prob_n, adv_n,
                   true_value_n):

        with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
            dist = self.get_distributions(obs_no)
            # Need to recompute this to record the gradient in the gradient tape
            pred_value_n = self.critic(obs_no)

            new_log_prob_n = dist.log_prob(ac_na)
            entropies = dist.entropy()
            # print(new_log_probs)
            # print(entropies)
            act_loss = self.actor_loss(new_log_prob_n, log_prob_n, entropies, adv_n)
            crt_loss = self.critic_loss(true_value_n, pred_value_n)

        gradients_of_actor = act_tape.gradient(act_loss, self.actor.trainable_variables)
        gradients_of_critic = crt_tape.gradient(crt_loss, self.critic.trainable_variables)

        self.actor_opt.apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))


    def run_epoch(self, obs, ac, log_prob, t_val, adv,
                  epochs, summary_writer, rewards, ite, batch_size=64):
        size = len(obs)
        train_indicies = np.arange(size)

        for epoch in range(epochs):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]

                self.train_step(obs[idx, :], ac[idx, :], log_prob[idx], adv[idx], t_val[idx])

                self.reward_metric(np.array(rewards).mean())
                # break
            tf.summary.scalar('reward', self.reward_metric.result(), step=epochs*ite+epoch)
            self.reward_metric.reset_states()
