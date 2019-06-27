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
        self.optimizers = [ self.actor_opt, self.critic_opt ]


    def act_stochastic(self, obs):
        action, _, log_prob, _ = self.actor(obs[None], training=True)
        value = self.critic(obs[None])

        return action[0], value[0], log_prob[0]


    def act_deterministic(self, obs):
        action, _, _, _ = self.actor.predict(obs)

        return action
        

    def actor_loss(self, new_log_probs, old_log_probs, entropy, advs):
        ratio = tf.exp(new_log_probs - old_log_probs)
        tf.print(tf.reduce_mean(ratio))
        clipped_ratio = tf.clip_by_value(ratio,
                                         1.0 - self.epsilon,
                                         1.0 + self.epsilon)
        surrogate_min = tf.minimum(ratio*advs, clipped_ratio*advs)
        surrogate_loss = tf.reduce_mean(surrogate_min)
        entropy_loss = tf.reduce_mean(entropy)
        
        return surrogate_loss + entropy_loss


    def critic_loss(self, values, target_values):
        value_loss = tf.keras.losses.mean_squared_error(y_true=target_values, y_pred=values)
        value_loss = tf.reduce_mean(value_loss)

        return value_loss


    @tf.function
    def train_step(self, observations, target_values, advantages, old_log_probs):
        
        with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
            actions, logits, log_probs, entropies = self.actor(observations, training=True)
            values = self.critic(observations) # recomputing just for clarity

            act_loss = self.actor_loss(log_probs, old_log_probs, entropies, advantages)
            crt_loss = self.critic_loss(values, target_values)

        gradients_of_actor = act_tape.gradient(act_loss, self.actor.trainable_variables)
        gradients_of_critic = crt_tape.gradient(crt_loss, self.critic.trainable_variables)

        self.optimizers[0].apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        self.optimizers[1].apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))


    def run_epoch(self, obs, old_log_probs, t_val, adv,
                  epochs, batch_size=64):
        size = len(obs)
        train_indicies = np.arange(size)

        for _ in range(epochs):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]

                self.train_step(obs[idx, :], t_val[idx], adv[idx], old_log_probs[idx])

