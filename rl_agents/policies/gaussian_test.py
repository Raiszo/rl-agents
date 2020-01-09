from rl_agents.policies.gaussian import GaussianActor, ContinuousSample
from tensorflow_probability import distributions

import tensorflow as tf
import numpy as np

class LayerTest(tf.test.TestCase):

    def setUp(self):
        super(LayerTest, self).setUp()
        self.layer = ContinuousSample((3,))

    def testShape(self):
        x = np.array([
            [0.5, 1.3, 3.6],
            [1.0, 2.0, 0.1],
        ]).astype(np.float32)

        pi, logp_pi, dist, inputs = self.layer(x)
        # print(inputs, x)

        self.assertShapeEqual(np.zeros((2,3)), pi) # should be 0s and 1s
        self.assertShapeEqual(np.zeros((2,)), logp_pi)
        self.assertIsInstance(dist, distributions.Normal)
        # self.assertAllEqual(x, inputs)

if __name__ == '__main__':
    tf.test.main()
