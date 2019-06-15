import numpy as np


def rollouts_generator(agent, env, horizon):
    """
    Generator function
    This function will continue generating
    samples as long as __next__() method is called
    """
    t = 0
    ac = env.action_space.sample()
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    
    new = True

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    log_probs = np.array([ac for _ in range(horizon)])
    vpreds = np.zeros(horizon, 'float32')

    news = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float32')

    while True:
        # prevac = ac
        # ac, vpred = pi.act(ob)
        
        ac, _, _, _ = agent.act(ob, sess)
        # print(ac)
        """
        Need next_vpred if the batch ends in the middle of an episode, then we need to append
        that value to vpreds to calculate the target Value using TD => V = r + gamma*V_{t+1}
        Else (finished episode) then append justa 0, does not mean that the value is 0
        but the Value target for the last step(T-1) is just the reward => V = r
        """
        if t > 0 and t % horizon == 0:
            yield { "ob": obs, "ac": acs, "rew": rews, "new": news,
                    "vpred": vpreds, "next_vpred": vpred*(1-new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                    "log_probs": log_probs }
            ep_rets = []
            ep_lens = []
        
        i = t % horizon

        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        log_probs[i] = log_prob
        news[i] = new

        ob, rew, new, _ = env.step(ac)
        # print(rew)
        rew = np.sum(rew)
        # print(ob, rew)

        rews[i] = rew
        cur_ep_ret += rew
        cur_ep_len += 1

        # if new or (ep_len and i > ep_len):
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

        t += 1

def add_vtarg_adv(seg, lam, gamma):
    T = len(seg["ob"])
    new = np.append(seg["new"], 0)
    seg["adv"] = gae_adv = np.empty(T, 'float32')
    seg["vtarg"] = td_v = np.empty(T, 'float32')
    
    vpred = np.append(seg["vpred"], seg["next_vpred"])

    last_gae = 0
    for t in reversed(range(T)):
        # check this, when is_terminal = 1-new[t], everything crushes like crazy
        is_terminal = 1-new[t+1]
        delta = - vpred[t] + (is_terminal * gamma * vpred[t+1] + seg["rew"][t])
        gae_adv[t] = last_gae = delta + gamma*lam*last_gae*is_terminal

        td_v[t] = is_terminal * gamma * vpred[t+1] + seg["rew"][t]
