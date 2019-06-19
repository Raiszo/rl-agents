import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as dists
import tensorflow.keras.layers as kl


class ContinuousSample(kl.Layer):
    def __init__(self, action_dim, log_std=-0.53):
        super(ContinuousSample, self).__init__(name='ContinuousSample')

        s_init = tf.constant_initializer(np.exp(log_std))
        self.std = tf.Variable(initial_value=s_init(shape=(1, action_dim),
                                                    dtype='float32'),
                               trainable=True)

    def call(self, logits, training):
        if not training:
            return logits, None, None, None
        else:
            distribution = dists.Normal(loc=logits, scale=self.std)

            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)
            entropy = distribution.entropy()

            return sample, logits, log_prob, entropy


class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, is_continuous,
                 size=32, num_layers=2):
        super(Actor, self).__init__(name='Actor')
        self.continuous = is_continuous

        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.nn.relu)
        self.layer_2 = kl.Dense(size, activation=tf.nn.relu)
        
        # Logits
        self.logits = kl.Dense(act_dim[0])
        # Sample
        self.sample = ContinuousSample(act_dim[0])
        
        
    def call(self, inputs, training=False):
        # x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.logits(x)

        if training:
            x = self.sample(logits, training=training)
        
        return x


class Critic(tf.keras.Model):
    def __init__(self, obs_dim,
                 size=32, num_layers=2):
        super(Critic, self).__init__(name='Critic')
        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.nn.relu)
        self.layer_2 = kl.Dense(size, activation=tf.nn.relu)
        
        # Logits
        self.value = kl.Dense(0)
        
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        return self.value(x)
