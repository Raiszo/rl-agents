from typing import Any, List, Sequence, Tuple, Callable, NoReturn
import datetime
from os.path import join

import tqdm
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
def get_env(env_name: str = 'Pendulum-v0') -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    return env

# Overview
# - DONE environment setup
# - DONE actor - critic model
# - DONE run environment with model
# - DONE rollout loop
# - DONE compute expected return
# - DONE compute loss
# - DONE trainning step
# - trainning loop
# - logging?
# - model saving?


class GaussianSample(layers.Layer):
    """
    Custom keras layer that implements a diagonal Gaussian distribution layer
    that stores the log_std
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
       """
       input_shape: might be [None, act_dim]
       """
       self.log_std = self.add_weight(
           'log_std', initializer=tf.keras.initializers.Zeros(),
           # 'log_std', initializer=tf.keras.initializers.Constant(-0.53),
           shape=(input_shape[1],), trainable=True
       )
       self.normal_dist = tfpl.DistributionLambda(
           make_distribution_fn=lambda x_std: tfpd.Normal(loc=x_std[0], scale=x_std[1]),
           convert_to_tensor_fn=lambda s: s.sample(),
       )

    def call(self, input):
        # better to pass the all values here, remember lambdas should be pure
        return self.normal_dist((input, tf.exp(self.log_std)))

def get_actor(obs_dim: int, act_dim: int) -> tf.keras.Model:
    """Get an actor stochastic policy"""
    observation = tf.keras.Input(shape=(obs_dim,))
    x = layers.Dense(64, activation='tanh')(observation)
    x = layers.Dense(64, activation='tanh')(x)
    logits = layers.Dense(act_dim, name='logits')(x)
    distributions = GaussianSample(name='gaussian_sample')(logits)

    actor = tf.keras.Model(observation, distributions)
    # actor.summary()

    return actor

def get_critic(obs_dim: int) -> tf.keras.Model:
    """Get a critic that returns the expect value for the current state"""
    observation = tf.keras.Input(shape=(obs_dim,), name='observation')
    x = layers.Dense(64, activation='tanh')(observation)
    x = layers.Dense(64, activation='tanh')(x)
    value = layers.Dense(1, name='value')(x)

    critic = tf.keras.Model(observation, value)
    # critic.summary()

    return critic

def get_model(obs_dim: int, act_dim: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    actor = get_actor(obs_dim, act_dim)
    critic = get_critic(obs_dim)

    return actor, critic


TFStep = Callable[[tf.Tensor], List[tf.Tensor]]

def get_env_step(env: gym.Env) -> TFStep:
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

    def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(env_step, [action],
                                 [tf.float32, tf.float32, tf.int32])

    return tf_env_step

def run_rollout(
        env: gym.Env,
        env_step: TFStep,
        initial_state: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        max_steps: int) -> List[tf.Tensor]:
    """Run the model in the environment for t=max_steps"""

    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # print('initial state')
    initial_state_shape = initial_state.shape
    state = initial_state
    # state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action "probabilities" and critic value
        action_dists, value_n = actor(state), critic(state)
        action_na = action_dists.sample()

        # get log probabilities by summing the logs, remember action is [n x a]
        logp_n = action_dists.log_prob(action_na)
        logp_n = tf.reduce_sum(logp_n, axis=1)

        # Store critic values
        values = values.write(t, tf.squeeze(value_n))

        # Store log probability of the action chosen
        action_log_probs = action_log_probs.write(t, tf.squeeze(logp_n))

        # Apply action to the environment to get next state and reward
        # be careful when squeezing, it may drop it to a single scaler if [1,1,1] :v
        state, reward, done = env_step(tf.squeeze(action_na, axis=[0]))
        # tf.ensure_shape(state, initial_state_shape)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_log_probs = action_log_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    # these are simple arrays
    return action_log_probs, values, rewards

def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        # not sure about this one
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + eps))

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
        action_log_probs_n: tf.Tensor,
        values_n: tf.Tensor,
        returns_n: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage_n = returns_n - values_n
    advantage_n = ((advantage_n - tf.math.reduce_mean(advantage_n)) /
                   (tf.math.reduce_std(advantage_n) + eps))

    actor_loss = -tf.math.reduce_sum(action_log_probs_n * advantage_n)

    critic_loss = huber_loss(values_n, returns_n)

    return actor_loss + critic_loss


actor_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
critic_opt = tf.keras.optimizers.Adam(learning_rate=3e-5)


# this speeds like 15x, do not forget to add
@tf.function
def train_step(
        env: gym.Env,
        env_step: TFStep,
        initial_state: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:

        # Run the model for one episode to collect training data
        action_log_probs_n, values_n, rewards_n = run_rollout(
            env, env_step, initial_state, actor, critic, max_steps_per_episode)

        # Calculate expected returns
        returns_n = get_expected_return(rewards_n, gamma, standardize=False)

        # Convert training data to appropriate TF tensor shapes
        action_log_probs_n, values_n, returns_n = [
            tf.expand_dims(x, 1) for x in [action_log_probs_n, values_n, returns_n]]

        # Calculating loss values to update our network
        loss = compute_loss(action_log_probs_n, values_n, returns_n)

    # Compute the gradients from the loss
    actor_grads = act_tape.gradient(loss, actor.trainable_variables)
    critic_grads = crt_tape.gradient(loss, critic.trainable_variables)

    # Apply the gradients to the model's parameters
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards_n)
    # print(actor.trainable_variables)

    return episode_reward

#####
# Shmall animation
#####

def render_episode(env: gym.Env, actor: tf.keras.Model, max_steps: int) -> NoReturn:
    screen = env.render(mode='rgb_array')

    state = tf.constant(env.reset(), dtype=tf.float32)
    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_na = actor(state).sample()

        state, _, done, _ = env.step(action_na[0])
        state = tf.constant(state, dtype=tf.float32)

        # Render screen every 10 steps
        if i % 10 == 0:
            screen = env.render(mode='rgb_array')

        if done:
            break

#####
# Trainning loop
#####

if __name__ == '__main__':

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(join('logs', current_time))

    max_episodes = 40000
    max_steps_per_episode = 200
    # max_episodes = 1
    # max_steps_per_episode = 4

    # Pendulum-v0 is considered solved if average reward is >= 180 over 100
    # consecutive trials
    # some benchmarks here: https://github.com/gouxiangchen/ac-ppo
    reward_threshold = -300
    running_reward = 0

    # Discount factor for future rewards
    gamma = 0.99

    env = get_env('Pendulum-v0')
    env_step = get_env_step(env)

    # Only for continuous obs & continuous act
    assert isinstance(env.observation_space, Box) and \
        isinstance(env.action_space, Box)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    actor = get_actor(obs_dim, act_dim)
    critic = get_critic(obs_dim)

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)
            episode_reward = int(train_step(
                env, env_step, initial_state, actor, critic,
                actor_optimizer=actor_opt, critic_optimizer=critic_opt,
                gamma=gamma, max_steps_per_episode=max_steps_per_episode))

            running_reward = episode_reward*0.01 + running_reward*.99

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if i % 10 == 0:
                pass # print(f'Episode {i}: average reward: {avg_reward}')

            with writer.as_default():
                tf.summary.scalar('log std', actor.get_layer('gaussian_sample').log_std[0], i)
                tf.summary.scalar('epoch mean', episode_reward, i)

            # finish if running_reward is better than threshold and if
            # episodes are greater than the running_window steps
            # important the second part when working with negative rewards
            if running_reward > reward_threshold and i >= 100:
                break

        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

        render_episode(env, actor, 200)
