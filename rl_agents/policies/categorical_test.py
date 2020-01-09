from rl_agents.policies.categorical import CategoricalActor, CategoricalSample
from tensorflow_probability import distributions

import tensorflow as tf
import numpy as np


class LayerTest(tf.test.TestCase):

    def setUp(self):
        super(LayerTest, self).setUp()
        self.layer = CategoricalSample()

    def testShape(self):
        x = np.array([
            [0.5, 1.3, 3.6],
            [1.0, 2.0, 0.1],
        ]).astype(np.float32)

        pi, logp_pi, dist, inputs = self.layer(x)

        self.assertShapeEqual(np.zeros((2,3)), pi)
        self.assertShapeEqual(np.zeros((2,)), logp_pi)
        self.assertIsInstance(dist, distributions.Multinomial)
        self.assertAllEqual(inputs, x)

        # test if one hot
        self.assertAllEqual(tf.reduce_sum(pi, axis=1), np.ones(x.shape[0]))

if __name__ == '__main__':
    tf.test.main()
