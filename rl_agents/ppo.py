from typing import Any, List, Sequence, Tuple, Callable, NoReturn, Dict, Optional
import datetime
from os import path
import uuid

import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfpd, layers as tfpl
from tensorboard.plugins.hparams import api as hp
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
# - DONE trainning loop
# - DONE logging?
# - DONE model saving?
# - DONE rollout loop could restart and get more data
# - DONE minibatch traing
# - DONE multiple epochs
# - get_expected_return should return list of the multiple env resets


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
           # 'log_std', initializer=tf.keras.initializers.Zeros(),
           # 'log_std', initializer=tf.keras.initializers.Constant(-0.53),
           'log_std', initializer=tf.keras.initializers.Constant(0.4),
           shape=(input_shape[1],), trainable=True
       )
       self.normal_dist = tfpl.DistributionLambda(
           make_distribution_fn=lambda x_std: tfpd.Normal(loc=x_std[0], scale=x_std[1]),
           convert_to_tensor_fn=lambda s: s.sample(),
       )

    def call(self, input):
        # better to pass the all values here, remember lambdas should be pure
        return self.normal_dist((input, tf.exp(self.log_std)))

    # since no additional variable is used in the __init__ function, no need to declare this
    # def get_config(self):
    #     return super().get_config()

def get_actor(obs_dim: int, act_dim: int, output_activation: str) -> tf.keras.Model:
    """Get an actor stochastic policy"""
    observation = tf.keras.Input(shape=(obs_dim,))
    x = layers.Dense(64, activation='tanh')(observation)
    x = layers.Dense(64, activation='tanh')(x)
    logits = layers.Dense(act_dim, activation=output_activation, name='logits')(x)
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

def get_model(obs_dim: int, act_dim: int, actor_output_activation: str) -> Tuple[tf.keras.Model, tf.keras.Model]:
    actor = get_actor(obs_dim, act_dim, actor_output_activation)
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

def get_env_reset(env: gym.Env) -> Callable[[], tf.Tensor]:
    """
    Return a Tensorflow function that wraps OpenAI Gym's `env.restart` call
    This would allow it to be included in a callable TensorFlow graph.
    """
    def env_reset() -> np.ndarray:
        """
        returns new state, reward, done
        """
        state = env.reset()
        return state.astype(np.float32)

    def tf_env_reset() -> tf.Tensor:
        # return tf.numpy_function(env_reset, [], tf.float32)
        return tf.numpy_function(env_reset, [], tf.float32)

    return tf_env_reset

def run_rollout(
        env: gym.Env,
        env_step: TFStep,
        env_reset: TFStep,
        initial_state: tf.Tensor,
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        steps: int) -> List[tf.Tensor]:
    """Run the model in the environment for t=max_steps"""

    action_log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    reward_sums = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state
    # state = initial_state

    # first episode
    j = 0
    reward_sums = reward_sums.write(j, 0.0)
    for t in tf.range(steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action "probabilities" and critic value
        action_dists, value_n = actor(state), critic(state)
        action_na = action_dists.sample()

        # store action values, be carefull squeezing when action space is only 1-dim
        actions = actions.write(t, tf.squeeze(action_na, axis=1))

        # get log probabilities by summing the logs, remember action is [n x a]
        logp_n = action_dists.log_prob(action_na)
        logp_n = tf.reduce_sum(logp_n, axis=1)

        # store state values
        states = states.write(t, tf.squeeze(state))

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
        # add the current reward to the cumulative reward for the episode
        reward_sums = reward_sums.write(j, reward_sums.read(j) + reward)

        # store dones
        dones = dones.write(t, done)

        if tf.cast(done, tf.bool):
            state = env_reset()
            # need this for @tf.function
            state.set_shape(initial_state_shape)
            # the episode is completed
            j += 1
            reward_sums = reward_sums.write(j, 0.0)


    action_log_probs = action_log_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    states = states.stack()
    actions = actions.stack()
    dones = dones.stack()
    reward_sums = reward_sums.stack()


    # these are simple arrays
    return actions, states, dones, values, rewards, action_log_probs, reward_sums

def get_expected_return(
        rewards_n: tf.Tensor,
        dones_n: tf.Tensor,
        gamma: float) -> tf.Tensor:
    """
    Compute expected returns per timestep.

    1 2 3 4 5(done) 6 7 8(done) 9 10
    10 9 8(done) 7 6 5(done) 4 3 2 1
    """
    n = tf.shape(rewards_n)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards_n` and accumulate reward sums
    # into the `returns` array
    rewards_n = tf.cast(rewards_n[::-1], dtype=tf.float32) # sometimes int32
    dones_n = dones_n[::-1]                                # always int32
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape

    for i in tf.range(n):
        reward = rewards_n[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

        # if current is done, reset discounted sum
        if i < n-1 and tf.cast(dones_n[i+1], tf.bool):
            discounted_sum = tf.constant(0.0)

    returns = returns.stack()[::-1]

    return returns

def compute_loss(
        critic_loss_fn: tf.keras.losses.Loss, # bad type hint :c
        actor: tf.keras.Model,
        critic: tf.keras.Model,
        actions_na: tf.Tensor,
        states_no: tf.Tensor,
        old_log_probs_n1: tf.Tensor,
        returns_n1: tf.Tensor,
        epsilon=0.2) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    # doing the forward pass
    action_dists, values_n1 = actor(states_no), critic(states_no)

    adv_n1 = returns_n1 - values_n1
    adv_n1 = ((adv_n1 - tf.math.reduce_mean(adv_n1)) /
              (tf.math.reduce_std(adv_n1) + eps))
    # remember if action is countinuous and space > 1, need to sum the log_props
    log_probs_n1 = tf.reduce_mean(action_dists.log_prob(actions_na), axis=1, keepdims=True)
    ratio_n = tf.exp(log_probs_n1 - old_log_probs_n1)
    clipped_ratio_n = tf.clip_by_value(ratio_n, 1.0 - epsilon, 1.0 + epsilon)


    surrogate_min = tf.minimum(ratio_n * adv_n1, clipped_ratio_n * adv_n1)
    surrogate_min = tf.reduce_mean(surrogate_min)
    critic_loss = critic_loss_fn(returns_n1, values_n1)

    return -surrogate_min + 0.5*critic_loss - 0.01*tf.reduce_mean(action_dists.entropy())

TFStep = Callable[[tf.Tensor], List[tf.Tensor]]

def get_train_step(
        env: gym.Env,
        env_step: TFStep,
        env_reset: TFStep,
        actor: tf.keras.Model, old_actor: tf.keras.Model,
        critic: tf.keras.Model,
        critic_loss_fn: tf.keras.losses.Loss,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        gamma: float,
        n_epochs: int,
        minibatch_size: int,
        iteration_size: int) -> Callable[[tf.Tensor], tf.Tensor]:

    @tf.function
    def train_step(initial_state: tf.Tensor) -> tf.Tensor:
        """Runs a model training step."""

        # Run the model for T=max_steps_per_iteration to collect training data using old_actor
        actions_na, states_no, dones_n, values_n, rewards_n, old_log_probs_n, reward_sums = run_rollout(
            env, env_step, env_reset, initial_state, old_actor, critic, iteration_size)

        # Calculate expected returns
        # print('before get_expected_return', rewards_n)
        returns_n = get_expected_return(rewards_n, dones_n, gamma)

        # Convert training data to appropriate TF tensor shapes
        old_log_probs_n1, values_n1, returns_n1, dones_n1 = [
            tf.expand_dims(x, 1) for x in [old_log_probs_n, values_n, returns_n, dones_n]]

        ds = tf.data.Dataset.from_tensor_slices((actions_na, states_no, old_log_probs_n1, returns_n1))
        ds = ds.shuffle(512).batch(minibatch_size).repeat(n_epochs)

        for actions, states, old_log_probs, returns in ds:
            # inside the tape actor & critic mus do a fordward computation
            # this fordward pass was done before in the rollout function, but, since
            # now the old_actor is the one running doing it, need to redo it inside compute_loss
            # print(actions, states, old_log_probs, returns)
            with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
                # Calculating loss values to update our network
                loss = compute_loss(critic_loss_fn, actor, critic, actions, states, old_log_probs, returns)

            # Compute the gradients from the loss
            actor_grads = act_tape.gradient(loss, actor.trainable_variables)
            critic_grads = crt_tape.gradient(loss, critic.trainable_variables)

            # Apply the gradients to the model's parameters
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        # print(reward_sums)
        # rewards_sums is shape (e,), where e is the number of env restarts + 1
        iteration_reward = tf.math.reduce_mean(reward_sums)

        return iteration_reward


    return train_step

#####
# Shmall animation
#####

def render_episode(env: gym.Env, actor: tf.keras.Model, max_steps: int) -> float:
    state = tf.constant(env.reset(), dtype=tf.float32)

    screen = env.render(mode='rgb_array')
    reward_sum = 0.0

    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_na = actor(state).mean()

        state, reward, done, _ = env.step(action_na[0])
        reward_sum += reward.astype(np.float32).item()
        state = tf.constant(state, dtype=tf.float32)

        screen = env.render(mode='rgb_array')

        if done:
            break

    return reward_sum

####
# Hparams
####
HP_N_ITERATIONS = hp.HParam('n_iterations')
HP_ITERATION_SIZE = hp.HParam('iteration_size')
HP_N_EPOCHS = hp.HParam('n_epochs')
HP_MINIBATCH_SIZE = hp.HParam('minibatch_size')
HP_ACTOR_LR = hp.HParam('actor_lr')
HP_CRITIC_LR = hp.HParam('critic_lr')

HP_GAMMA = hp.HParam('gamma')
HP_ENVIRONMENT = hp.HParam('environment')
HP_ACTOR_OUTPUT = hp.HParam('actor_output')
# HP_INITIAL_LOG_STD = hp.HParam('initial_log_std')
METRIC_FINAL_REWARD = 'final_reward'
METRIC_EPOCH_REWARD = 'epoch_reward'


def run_experiment(
        environment: str,
        n_iterations: int, iteration_size: int,
        n_epochs: int, minibatch_size: int,
        gamma: float,
        actor_lr: float, critic_lr: float,
        actor_output_activation: str,
        base_dir: str,
        early_stop_reward_threshold: Optional[int]=None) -> None:

    hparams = {
        HP_N_ITERATIONS: n_iterations,
        HP_ITERATION_SIZE: iteration_size,
        HP_N_EPOCHS: n_epochs,
        HP_MINIBATCH_SIZE: minibatch_size,

        HP_GAMMA: gamma,
        HP_ENVIRONMENT: environment,

        HP_ACTOR_LR: actor_lr,
        HP_CRITIC_LR: critic_lr,
        HP_ACTOR_OUTPUT: actor_output_activation,
        # HP_INITIAL_LOG_STD
    }
    trial_id = str(uuid.uuid4())

    save_hparams(hparams, path.join(base_dir, 'hparam_tuning'))


    # environment setup
    env = get_env(hparams[HP_ENVIRONMENT])
    env_step = get_env_step(env)
    env_reset = get_env_reset(env)

    actor_opt = tf.keras.optimizers.Adam(learning_rate=hparams[HP_ACTOR_LR])
    critic_opt = tf.keras.optimizers.Adam(learning_rate=hparams[HP_CRITIC_LR])

    # Only for continuous obs & continuous act
    assert isinstance(env.observation_space, Box) and \
        isinstance(env.action_space, Box)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    actor, critic = get_model(obs_dim, act_dim, actor_output_activation=hparams[HP_ACTOR_OUTPUT])
    old_actor = get_actor(obs_dim, act_dim, output_activation=actor_output_activation)

    # loss
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    # Discount factor for future rewards
    gamma = 0.99

    train_step = get_train_step(
        env, env_step, env_reset,
        actor=actor, old_actor=old_actor, critic=critic,
        critic_loss_fn=mse_loss,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        gamma=hparams[HP_GAMMA],
        n_epochs=hparams[HP_N_EPOCHS],
        minibatch_size=hparams[HP_MINIBATCH_SIZE],
        iteration_size=hparams[HP_ITERATION_SIZE])

    # tb_callback = tf.keras.callbacks.TensorBoard(logdir)
    # tb_callback.set_model(actor)

    trial_dir = path.join(base_dir, 'trials', trial_id)
    # {trial_id}/logs
    writer = tf.summary.create_file_writer(path.join(trial_dir, 'logs'))

    running_reward = 0
    with tqdm.trange(hparams[HP_N_ITERATIONS]) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=tf.float32)

            # tf.summary.trace_on(graph=True, profiler=True)
            iteration_reward = int(train_step(initial_state))
            # with writer.as_default():
            with writer.as_default():
                # tf.summary.trace_export(
                #     name='train_step',
                #     step=i,
                #     profiler_outdir=logdir)
                tf.summary.scalar('log std', actor.get_layer('gaussian_sample').log_std[0], i)
                tf.summary.scalar(METRIC_EPOCH_REWARD, iteration_reward, i)

            # update old-actor with the learned actor
            old_actor.set_weights(actor.get_weights())


            running_reward = iteration_reward*0.01 + running_reward*.99

            t.set_description(f'Iteration {i}')
            t.set_postfix(
                iteration_reward=iteration_reward, running_reward=running_reward)

            # save each 50 steps
            if i % 50 == 0:
                # {trial_id}/models/[actor|critic]
                actor.save(path.join(trial_dir, 'models', f'actor_{i}'), save_traces=False)
                critic.save(path.join(trial_dir, 'models', f'critic_{i}'), save_traces=False)

            # finish if running_reward is better than threshold and if
            # number of iterations are greater than the running_window steps
            # important the second part when working with negative rewards
            if early_stop_reward_threshold and running_reward > early_stop_reward_threshold and i >= 100:
                break

        # save the final model
        actor.save(path.join(trial_dir, 'models', f'actor_{t.n}'), save_traces=False)
        critic.save(path.join(trial_dir, 'models', f'critic_{t.n}'), save_traces=False)

        print(f'\nSolved at iteration {i}: average reward: {running_reward:.2f}!')

        final_reward_mean = render_episode(env, actor, 200)

        with writer.as_default():
            hp.hparams(hparams, trial_id=trial_id)
            tf.summary.scalar(METRIC_FINAL_REWARD, final_reward_mean ,step=1)

def save_hparams(hparams: Dict[hp.HParam, Any], hparams_dir: str) -> None:
    # saved in experiments/hparam_tuning
    with tf.summary.create_file_writer(hparams_dir).as_default():
        hp.hparams_config(
            hparams=[HP_N_ITERATIONS, HP_ITERATION_SIZE, HP_N_EPOCHS, HP_MINIBATCH_SIZE,
                     HP_GAMMA, HP_ENVIRONMENT,
                     HP_ACTOR_LR, HP_CRITIC_LR, HP_ACTOR_OUTPUT],
            metrics=[hp.Metric(METRIC_FINAL_REWARD, display_name='final reward mean'),
                     hp.Metric(METRIC_EPOCH_REWARD, display_name='epoch reward')],
        )


if __name__ == '__main__':
    # Pendulum-v0 is considered solved if average reward is >= 180 over 100
    # consecutive trials
    # some benchmarks here: https://github.com/gouxiangchen/ac-ppo
    base_dir = 'experiments'

    ####
    # Experiment parameters
    ####
    # base dir is experiments/trials
    run_experiment(
        environment='Pendulum-v0',
        n_iterations=600, iteration_size=2048,
        n_epochs=10, minibatch_size=64,
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=5e-3,
        actor_output_activation='linear',
        base_dir=base_dir,
        early_stop_reward_threshold=-200
    )
