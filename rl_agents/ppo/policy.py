import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as dists
import tensorflow.keras.layers as kl


class ContinuousSample(kl.Layer):
    def __init__(self, action_dim, log_std=-0.53):
        super().__init__(name='ContinuousSample')

        s_init = tf.constant_initializer(np.exp(log_std))
        self.std = tf.Variable(initial_value=s_init(shape=(1, action_dim),
                                                    dtype='float32'),
                               trainable=True)

    def call(self, logits, training):
        # No need to add zeros lie logis, this shit should be broadcasted
        if not trainning:
            return logits, None, None
        
        distribution = dists.Normal(loc=logits, scale=self.std)
        
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)

        return sample, log_prob, distribution, self.std, logits
        

class Actor(tf.keras.Model):
    def __init__(self, obs_space, act_space, is_continuous, size=32, num_layers=2):
        super().__init__(name='Actor')
        self.continuous = is_continuous

        self.layer_1 = kl.Dense(size, input_shape=obs_space, activation=tf.nn.relu)
        self.layer_2 = kl.Dense(size, activation=tf.nn.relu)
        
        # Logits
        self.logits = kl.Dense(act_space[0])

        # Sample
        self.sample = ContinuousSample(act_space[0])
        
        
    def call(self, inputs):
        # x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        logits = self.logits(x)
        sample = self.sample(logits)

        return sample

    def action(self, obs):
        res = self.predict(obs)
        print(res)

        return res

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
