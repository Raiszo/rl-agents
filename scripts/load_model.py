import numpy as np
import tensorflow as tf
from rl_agents.ppo import GaussianSample, get_env

#####
# Shmall animation
#####

def run_episode(env: gym.Env, actor: tf.keras.Model, max_steps: int) -> float:
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



if __name__ == '__main__':
    actor_dir = 'experiments/trials/6f4a3efe-f6c3-4126-8689-2e3d5babbca7/models/actor_300'
    env = get_env('Pendulum-v0')
    custom_objects={'GaussianSample': GaussianSample}
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(actor_dir)

        run_episode(env, actor, 200)
