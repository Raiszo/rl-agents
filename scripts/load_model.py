import tensorflow as tf
from rl_agents.ppo import GaussianSample, render_episode, get_env


if __name__ == '__main__':
    actor_dir = 'experiments/trials/6f4a3efe-f6c3-4126-8689-2e3d5babbca7/models/actor_300'
    env = get_env('Pendulum-v0')
    custom_objects={'GaussianSample': GaussianSample}
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(actor_dir)

        render_episode(env, actor, 200)
