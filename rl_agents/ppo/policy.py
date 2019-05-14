import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as dist
import tensorflow.keras.layers as kl


class Actor(tf.keras.Model):
    def __init__(self, obs_space, act_space, is_continuous):
        super().__init__(name='Actor')
        self.continuous = is_continuous
        
        self.layers = [kl.Dense(size, activation=tf.nn.relu) for _ in range num_layers]
        
        # Logits
        self.output = kl.Dense(output_size, name='value')

        log_std = -0.53 * tf.ones([1, tf.logits.shape[1]])
        self.std = tf.zeros_like(self.logits) + tf.exp(log_std)
        
    def call(self, inputs):
        hidden = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for l in self.layers():
            hidden = l(obs)
        logits = self.output(hidden)
            
        distribution = dist.Normal(loc=logits, scale=self.std)
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        
        return sample, log_prob

    

class Critic(tf.keras.Model):
    def __init__(self, obs_space):
        self.layers = [kl.Dense(size, activation=tf.nn.relu) for _ in range num_layers]
        
        # This is the value
        self.output = tf.squeeze()
        
    def call(self, inputs):
        hidden = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for l in self.layers():
            hidden = l(obs)
            
        return self.value(hidden)
    
class PPO:
    def __init__(self):
        self.a = 21
