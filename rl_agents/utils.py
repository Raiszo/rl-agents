from typing import Tuple
import tensorflow as tf
import numpy as np

@tf.function
def decaying_cumsum(a: tf.Tensor, factor: float) -> tf.Tensor:
    """
    - a: n dimensional tensor
    """
    n = tf.shape(a)[0]
    result = tf.TensorArray(dtype=tf.float32, size=n)

    decaying_sum = tf.constant(0.0)
    rev = a[::-1]

    for i in tf.range(n):
        decaying_sum = rev[i] + factor * decaying_sum
        result = result.write(i, decaying_sum)

    return result.stack()[::-1]


# @tf.function
def gae_advantage(rewards_n: tf.Tensor, dones_n: tf.Tensor, values_n: tf.Tensor,
                  last_value: tf.Tensor, gamma: float, lam: float) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(rewards_n)[0]

    adv = tf.TensorArray(dtype=tf.float32, size=n)
    dr = tf.TensorArray(dtype=tf.float32, size=n)

    r = tf.cast(rewards_n[::-1], dtype=tf.float32)
    v = values_n[::-1]
    d = tf.cast(dones_n[::-1], dtype=tf.float32)

    next_value = last_value
    delta_sum = tf.constant(0.0, dtype=tf.float32)
    delta_sum_shape = delta_sum.shape
    reward_sum = tf.constant(0.0, dtype=tf.float32)
    reward_sum_shape = reward_sum.shape
    for i in tf.range(n):
        # gae
        continues = 1-d[i]
        delta = r[i] + gamma * next_value * continues - v[i]
        next_value = v[i]
        delta_sum = gamma * lam * delta_sum * continues + delta
        delta_sum.set_shape(delta_sum_shape)
        # discounted reward
        reward_sum = r[i] + gamma * reward_sum
        reward_sum.set_shape(reward_sum_shape)

        adv = adv.write(i, delta_sum)
        dr = dr.write(i, reward_sum)

    return adv.stack()[::-1], dr.stack()[::-1]

# @tf.function
def simple_advantage(rewards_n: tf.Tensor, dones_n: tf.Tensor, values_n: tf.Tensor, gamma: float) -> Tuple[tf.Tensor, tf.Tensor]:
    n = tf.shape(rewards_n)[0]

    dr = tf.TensorArray(dtype=tf.float32, size=n)

    d = tf.cast(dones_n[::-1], dtype=tf.float32)
    r = tf.cast(rewards_n[::-1], dtype=tf.float32)

    reward_sum = tf.constant(0.0, dtype=tf.float32)
    reward_sum_shape = reward_sum.shape
    for i in tf.range(n):
        continues = 1-d[i]
        reward_sum = r[i] + gamma * reward_sum * continues
        reward_sum.set_shape(reward_sum_shape)

        dr = dr.write(i, reward_sum)

    discounted_rewards = dr.stack()[::-1]
    advs = discounted_rewards - values_n

    return advs, discounted_rewards

# TODO: tests
if __name__ == '__main__':
    a = tf.constant([1, 2, 3, 0, 2, -1, 5], dtype=tf.float32)
    dones = tf.constant([0, 0, 0, 1, 0, 1, 1], dtype=tf.int32)
    # dones = tf.constant([0, 0, 0, 1, 0, 1, 1], dtype=tf.int32)
    v = tf.constant([2, 5, 2, 0, -1, 10, 7], dtype=tf.float32)

    # res = decaying_cumsum(a, 0.99)
    # print(res)
    # adv, dr = gae_advantage(a, dones, v, tf.constant(8, dtype=tf.float32), 1.0, 1.0)
    # print(adv.numpy(), dr.numpy())

    adv, dr = simple_advantage(a, dones, v, 1.0)
    print(adv.numpy(), dr.numpy())
