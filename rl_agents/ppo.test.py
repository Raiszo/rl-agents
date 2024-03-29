# import unittest
import tensorflow as tf

from rl_agents import ppo

class GetExpectedReturnTest(tf.test.TestCase):
    def test_multiple_dones(self):
        rewards = tf.constant([1.0, 2.0, 3.0, 3.0, 2.0, 5.0], dtype=tf.float32)
        dones = tf.constant([0, 0, 1, 0, 1, 0], dtype=tf.int32)
        returns = ppo.get_expected_return(rewards, dones, gamma=1.0)

        self.assertAllClose(
            tf.constant([6.0, 5.0, 3.0, 5.0, 2.0, 5.0]),
            returns,
        )

    def test_no_done(self):
        rewards = tf.constant([1.0, 2.0, 3.0, -3.0, 2.0], dtype=tf.float32)
        dones = tf.constant([0, 0, 0, 0, 0], dtype=tf.int32)
        returns = ppo.get_expected_return(rewards, dones, gamma=1.0)

        self.assertAllClose(
            tf.constant([5.0, 4.0, 2.0, -1.0, 2.0]),
            returns,
        )


if __name__ == '__main__':
    tf.test.main()
