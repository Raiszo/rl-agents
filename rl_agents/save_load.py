import tensorflow as tf
from rl_agents.ppo import GaussianSample, get_actor, get_env, get_model

tf.random.set_seed(10)


def evaluate(model: tf.keras.Model) -> None:
    """
    Perform some computation to visually check if a model behaves the same way as another
    tfp.Distributions.Normal().sample: even when same mean/std are used a different result
    can be obtained
    """
    test = tf.constant([[2.3, 2.5]])
    dists = model(test)
    print(dists.sample())

# new model
actor = get_actor(2, 1, 'linear')
evaluate(actor)

# save it, if tfpl use save_traces to avoid warnings
actor.save('actor', save_format='tf', save_traces=False)


# load model, for tfpl layers, this seems mandatory
custom_objects={'GaussianSample': GaussianSample}
with tf.keras.utils.custom_object_scope(custom_objects):
    loaded_actor = tf.keras.models.load_model('actor')
    evaluate(loaded_actor)
