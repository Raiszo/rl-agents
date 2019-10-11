from abc import ABC, abstractmethod

def combined_shape(length, shape):
    return (length, shape) 
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class OnPolicyBuffer(ABC):
    @abstractmethod
    def get_advantage(self):
        pass

    def __init__(self, agent, env, is_continuous, horizon):
        self.agent = agent
        self.env = env
        self.is_continuous = is_continuous
        self.horizon = horizon

        self.t = 0

        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float64)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float64)
        self.logp_buf = np.zeros(size, dtype=np.float64)
        self.rew_buf = np.zeros(size, dtype=np.float64)
        self.val_buf = np.zeros(size, dtype=np.float64)
        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        ac = env.action_space.sample()
        if is_continuous:
            ac = ac.astype(np.float64)
        ob = env.reset().astype(np.float64)

        self.cur_ep_ret = 0 # return in current episode
        self.cur_ep_len = 0 # len of current episode
        self.ep_rets = [] # returns of completed episodes in this segment
        self.ep_lens = [] # lengths of ...
    
        self.new = True
