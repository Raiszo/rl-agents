import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions
import tensorflow.keras.layers as kl


class CategoricalSample(kl.Layer):
    def __init__(self):
        super(CategoricalSample, self).__init__(name='CategoricalSample')

    def call(self, inputs):
        dist = distributions.Multinomial(total_count=1.0, logits=inputs)

        pi = dist.sample()
        logp_pi = dist.log_prob(pi)
        pi = tf.argmax(pi, axis=1)

        return pi, logp_pi, dist, pi


class CategoricalActor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, size=32):
        super(CategoricalActor, self).__init__(name='Actor')

        self.layer_1 = kl.Dense(size, input_shape=obs_dim, activation=tf.keras.activations.tanh)
        self.layer_2 = kl.Dense(size, activation=tf.keras.activations.tanh)
        
        # Logits
        self.logits = kl.Dense(act_dim)
        # Sample
        self.sample = CategoricalSample()
        
        
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.logits(x)
        x = self.sample(x)

        return x
