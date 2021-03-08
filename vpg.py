from typing import Any, List, Sequence, Tuple, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfpd, layers as tfpl
import gym
from gym.spaces import Discrete, Box

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


# Create the environment
def get_env(env_name: str = 'MountainCarContinuous-v0') -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    return env

# Overview
# - DONE environment setup
# - DONE actor - critic model
# - run environment with model
# - rollout loop
# - compute expected return
# - compute loss
# - trainning step
# - trainning loop
# - logging?
# - model saving?


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
           'log_std', initializer=tf.keras.initializers.Zeros(),
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
    """Get an actor stochastic policy"""
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

def get_env_step(env: gym.Env) -> Callable[[tf.Tensor], List(tf.Tensor)]:
    """
    Return a Tensorflow function that wraps OpenAI Gym's `env.step` call
    This would allow it to be included in a callable TensorFlow graph.
    """
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

    return tf_env_step

def run_rollout(env: gym.Env, model: tf.keras.Model, max_steps: int) -> List[tf.Tensor]:
    """Run the model in the environment for t=max_steps"""
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

env = get_env()
