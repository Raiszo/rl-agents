from abc import ABC, abstractmethod

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

        ac = env.action_space.sample()
        if is_continuous:
            ac = ac.astype(np.float64)
        ob = env.reset().astype(np.float64)

        self.cur_ep_ret = 0 # return in current episode
        self.cur_ep_len = 0 # len of current episode
        self.ep_rets = [] # returns of completed episodes in this segment
        self.ep_lens = [] # lengths of ...
    
        self.new = True
