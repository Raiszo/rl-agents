import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as dists
import tensorflow.keras.layers as kl


class ContinuousSample(kl.Layer):
    def __init__(self, action_dim, log_std=-0.53):
        super(ContinuousSample, self).__init__(name='ContinuousSample')

        s_init = tf.constant_initializer(np.exp(log_std))
        self.std = tf.Variable(initial_value=s_init(shape=(1, action_dim),
                                                    dtype='float64'),
                               name='std',
                               trainable=True)

    def call(self, inputs, training):
        if training:
            std = tf.zeros_like(inputs) + self.std
            distribution = dists.Normal(loc=inputs, scale=std)

            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)

            return inputs, distribution, sample, log_prob
        else:
            return inputs, None, None, None


class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, is_continuous,
                 size=32, num_layers=2):
        super(Actor, self).__init__(name='Actor')
        self.continuous = is_continuous

        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.keras.activations.tanh)
        self.layer_2 = kl.Dense(size, activation=tf.keras.activations.tanh)
        
        # Logits
        self.logits = kl.Dense(act_dim[0])
        # Sample
        self.sample = ContinuousSample(act_dim[0])
        
        
    def call(self, inputs, training=False):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.logits(x)

        # if training:
        x = self.sample(x, training=training)
        
        return x


class Critic(tf.keras.Model):
    def __init__(self, obs_dim,
                 size=32, num_layers=2):
        super(Critic, self).__init__(name='Critic')
        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.keras.activations.tanh)
        self.layer_2 = kl.Dense(size, activation=tf.keras.activations.tanh)
        
        # Logits
        self.value = kl.Dense(1)
        
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.value(x)

        return x
