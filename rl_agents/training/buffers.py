from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import lfilter

def combined_shape(length, shape=None):
    # return (length, shape)
    # deprecated Conditional
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class OnPolicyBuffer:
    """
    A buffer for storing trajectories experienced by an On Policy agent interacting
    with the environment
    """

    def __init__(self, obs_dim, act_dim, size):
        # print(obs_dim, act_dim)
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.size = size

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def __len__(self):
        """
        So len(buff) works
        """
        return self.size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    @abstractmethod
    def get_advantage(self, rews, vals):
        pass

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # print(path_slice)
        # print('rews', rews)
        # print('vals', vals)
        # print('path slice', path_slice)
        self.adv_buf[path_slice], self.ret_buf[path_slice] = self.get_advantage(rews, vals)
        # print('ret+', self.ret_buf[path_slice])
        # print('adv+', self.adv_buf[path_slice])
        self.path_start_idx = self.ptr

    def get(self, ep_rets):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # advantage normalization trick
        # print(self.adv_buf)
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # print('get', self.adv_buf.mean(), self.adv_buf.std())

        return {
            "obs": self.obs_buf,
            "act": self.act_buf,
            "adv": self.adv_buf,
            "logp": self.logp_buf,
            "ret": self.ret_buf,
            "val": self.val_buf,
            "ep_rets": ep_rets,
        }

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class GAE_Buffer(OnPolicyBuffer):
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        super().__init__(obs_dim, act_dim, size)
        self.gamma = gamma
        self.lam = lam

    def get_advantage(self, rews, vals):
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        ret = discount_cumsum(rews, self.gamma)[:-1]

        return adv, ret
