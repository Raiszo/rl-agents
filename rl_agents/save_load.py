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
    # print('gaussian sample weight', model.get_layer('gaussian_sample').get_weights())
    # print('weights', model.get_weights())
    dists = model(test)
    print(dists.sample())

# new model
actor = get_actor(2, 1, 'linear')
evaluate(actor)

# save it
actor.save('actor')

# load model
loaded_actor = tf.keras.models.load_model(
    'actor',
    custom_objects={'GaussianSample': GaussianSample}
)
evaluate(loaded_actor)
