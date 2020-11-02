from typing import Any, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfpd, layers as tfpl
import gym
from gym.spaces import Discrete, Box

# Create the environment
env = gym.make('MountainCarContinuous-v0')

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
env.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class GaussianSample(tf.keras.Layer):
    """
    Custom keras layer that implements a diagonal Gaussian distribution layer
    that stores the log_std
    """
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
       """
       input_shape: might be [None, act_dim]
       """
       self.log_std = self.add_weight(
           'log_std', initializer=tf.keras.initializers.Constant(-0.53),
           shape=(input_shape[1],), trainable=True
       )
       self.normal_dist = tfpl.DistributionLambda(
           make_distribution_fn=lambda t: tfpd.Normal(loc=t, scale=tf.exp(self.log_std)),
           convert_to_tensor_fn=lambda s: s.sample(),
       )

    def call(self, input):
        #return tfpd.Normal(loc=input, scale=tf.exp(self.log_std))
        return self.normal_dist(input)


def get_actor(obs_dim: int, act_dim: int) -> tf.keras.Model:
    """
    Get an actor stochastic policy
    """
    mlp_input = tf.keras.Input(shape=(obs_dim,), name='x')
    x = layers.Dense(32, activation='tanh', name='dense_1')(mlp_input)
    mlp_output = layers.Dense(act_dim, name='logits')(x)
    mlp = tf.keras.Model(mlp_input, mlp_output)
    mlp.summary()

    sampler_input = tf.keras.Input(shape=(act_dim,), name='logits')
    sampler_output = GaussianSample(sampler_input, name='sample')
    sampler = tf.keras.Model(sampler_input, sampler_output)
    sampler.summary()

    observation = tf.keras.Input(shape=(obs_dim,), name='observation')
    logits = mlp(observation)
    action = sampler(logits)
    actor = tf.keras.Model(observation, action)

    return actor

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns new state, reward, done
    """
    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            reward.astype(np.float32), # maybe discrete envs only get discrete rewards
            np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List(tf.Tensor):
    return tf.numpy_function(env_step, [action],
                             [tf.float32, tf.float32, tf.int32])
