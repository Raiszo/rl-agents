import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl


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
