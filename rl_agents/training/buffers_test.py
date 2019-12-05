from rl_agents.training.buffers import GAE_Buffer

class TestGAEBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        self.buff = GAE_Buffer((3,), (1,), 20, gamma=0.99, lam=0.95)

    def test_reward(self):
        obs = np.array([1,2,3])
        ac = np.array([1])
        rew = -1
        val = 1
        logp = 2

        for i in range(10):
            self.buff.store(obs, ac, rew, val, logp)
        self.buff.finish_path(last_val=4)

        self.assertEqual()
