import tensorflow as tf
from tensorflow_probability import distributions as dist
import tensorflow.keras.layers as kl



class Actor(tf.keras.Model):
    def __init__(self, obs_space, act_space, is_continuous):
        super().__init__(name='Actor')
        self.layers = [kl.Dense(size, activation=tf.nn.relu) for _ in range num_layers]
        self.logits = kl.Dense(output_size, name='value')
        self.continuous = is_continuous
        self.distribution = dist.Normal(loc=self.logits )

        self.distribution = dist.Normal()
        
    def call(self, inputs):
        hidden = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for l in self.layers():
            hidden = l(obs)
            
        return self.logits(hidden)

class Critic(tf.keras.Model):
    def __init__(self, obs_space):
        self.layers = [kl.Dense(size, activation=tf.nn.relu) for _ in range num_layers]
        self.value = tf.squeeze()
        
    def call(self, inputs):
        hidden = tf.convert_to_tensor(inputs, dtype=tf.float32)
        for l in self.layers():
            hidden = l(obs)
            
        return self.value(hidden)
    
class PPO:
    def __init__(self):
        self.a = 21
