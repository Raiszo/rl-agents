import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions
import tensorflow.keras.layers as kl


class ContinuousSample(kl.Layer):
    def __init__(self, action_dim):
        super(ContinuousSample, self).__init__(name='ContinuousSample')

        # s_init = tf.constant_initializer(np.exp(log_std))
        log_std = -0.53 * np.ones(action_dim, dtype=np.float32)
        self.std = tf.Variable(initial_value=np.exp(log_std),
                               name='std', trainable=True)

    def call(self, inputs):
        # If training return dist, else not
        # So better to always return everything
        # std = tf.zeros_like(inputs) + self.std
        dist = distributions.Normal(loc=inputs, scale=self.std)

        pi = dist.sample()
        # this usually outputs an [N x act_dim] tensor,
        # thus one log prob per gaussian distribution mean/std
        # One needs just a single log prob value [Nx1] tensor
        logp_pi = dist.log_prob(pi)
        # https://www.reddit.com/r/reinforcementlearning/comments/bs1xw8/multi_dimensional_continuous_action_space/
        logp_pi = tf.reduce_sum(logp_pi, axis=1)

        return pi, logp_pi, dist, inputs


class GaussianActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, size=32, num_layers=2):
        super(GaussianActor, self).__init__(name='Actor')

        self.act_dim = act_dim

        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.keras.activations.tanh)
        self.layer_2 = kl.Dense(size, activation=tf.keras.activations.tanh)
        
        # Logits
        self.logits = kl.Dense(act_dim)
        # Sample
        self.sample = ContinuousSample(act_dim)
        
        
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.logits(x)
        x = self.sample(x)

        return x
