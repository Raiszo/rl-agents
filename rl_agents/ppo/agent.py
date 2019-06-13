import tensorflow as tf
from ppo.policy import Actor, Critic

class PPO_Agent(tf.keras.Model):
    def __init__(self,
                 actor,
                 critic,
                 epsilon=0.2):
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon

    def action_value(self, obs):
        action, log_prob, dist, logits = self.actor.predict(obs)
        value = self.critic.predict(obs)

        return action , value, log_prob, dist

    def loss(self, actions, log_probs, advantages, values, dist, target_values):
        ratio = tf.exp(dist.log_prob(actions) - log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon)


        surrogate_min = tf.minimum(ratio*advantages, clipped_ratio*advantages)
        surrogate_loss = tf.reduce_mean(surrogate_min)


        value_loss = tf.losses.mean_squared_error(labels=target_values, predictions=values)
        value_loss = tf.reduce_mean(value_loss)

        entropy_loss = tf.reduce_mean(dist.entropy())


    def train(self, env):
        losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

