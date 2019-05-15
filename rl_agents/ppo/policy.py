import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as dist
import tensorflow.keras.layers as kl


class Actor(tf.keras.Model):
    def __init__(self, obs_space, act_space, is_continuous, size=32, num_layers=2):
        super().__init__(name='Actor')
        self.continuous = is_continuous

        print(obs_space)
        self.layer_1 = kl.Dense(size, input_shape=obs_space, activation=tf.nn.relu)
        self.layer_2 = kl.Dense(size, activation=tf.nn.relu)
        
        # Logits
        self.logits = kl.Dense(act_space[0])
        
        self.log_std = log_std = -0.53 * tf.ones([1, act_space[0]])
        
    def call(self, inputs):
        # x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        logits = self.logits(x)

        return logits

    def action(self, obs):
        logits = self.predict(obs)
        print(logits)
        
        std = tf.zeros_like(logits) + tf.exp(self.log_std)
        
        distribution = dist.Normal(loc=logits, scale=std)
        sample = distribution.sample().numpy()
        log_prob = distribution.log_prob(sample).numpy()
        
        # return self.log_std
    
        return sample, log_prob
        # return self.predict(obs)
    

# class Critic(tf.keras.Model):
#     def __init__(self, obs_space):
#         self.layers = [kl.Dense(size, activation=tf.nn.relu) for _ in range num_layers]
        
#         # This is the value
#         self.output = tf.squeeze()
        
#     def call(self, inputs):
#         hidden = tf.convert_to_tensor(inputs, dtype=tf.float32)
#         for l in self.layers():
#             hidden = l(obs)
            
#         return self.value(hidden)
    
# class PPO:
#     def __init__(self):
#         self.a = 21
