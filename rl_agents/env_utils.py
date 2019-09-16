import numpy as np


def rollouts_generator(agent, env, is_continuous, horizon):
    """
    Generator function
    This function will continue generating
    samples as long as __next__() method is called
    """
    t = 0
    ac = env.action_space.sample()
    if is_continuous:
        ac = ac.astype(np.float64)
    ob = env.reset().astype(np.float64)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...
    
    new = True

    obs = np.array([ob for _ in range(horizon)])
    acs = np.array([ac for _ in range(horizon)])
    if is_continuous:
        log_probs = np.array([ac for _ in range(horizon)])
    else:
        log_probs = np.zeros(horizon, 'float64')
    vpreds = np.zeros(horizon, 'float64')

    news = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float64')

    while True:
        # if t % 500 == 0:
        #     print(agent.actor.trainable_variables[6])
        
        if t > 0 and t % horizon == 0:
            # When comparing this with the ones that are calculated in the train step
            # The first one is different, the others are the same
            # Without the first one everything works well
            _, _, vpred = agent.act_stochastic(ob)

            yield { "ob": obs, "ac": acs, "rew": rews, "new": news,
                    "vpred": vpreds, "next_vpred": vpred*(1-new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                    "log_probs": log_probs }
            ep_rets = []
            ep_lens = []
        
        i = t % horizon

        news[i] = new
        obs[i] = ob

        ac, log_prob, vpred = agent.act_stochastic(ob)
        """
        Need next_vpred if the batch ends in the middle of an episode, then we need to append
        that value to vpreds to calculate the target Value using TD => V = r + gamma*V_{t+1}
        Else (finished episode) then append justa 0, does not mean that the value is 0
        but the Value target for the last step(T-1) is just the reward => V = r
        """

        acs[i] = ac
        log_probs[i] = log_prob
        vpreds[i] = vpred

        # the ob and new here will be used in the next iteration of the while loop
        # print(ac.numpy())
        ob, rew, new, _ = env.step(ac.numpy())
        # print(rew)
        # rew = rew - 5
        # rew = np.sum(rew)
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

# def GAE(rewards, value, news):
    
def get_adv_vtarg(roll, lam, gamma):
    T = len(roll["ob"])
    new = np.append(roll["new"], 0)

    gae_adv = np.empty(T, 'float64')
    target_val = np.empty(T, 'float64')
    
    vpred = np.append(roll["vpred"], roll["next_vpred"])

    last_gae = 0
    for t in reversed(range(T)):
        # check this, when is_terminal = 1-new[t], everything crushes like crazy
        # If the next step is corresponds to a new iteration, then this is the end
        is_terminal = 1-new[t+1]
        delta = - vpred[t] + (is_terminal * gamma * vpred[t+1] + roll["rew"][t])
        gae_adv[t] = last_gae = delta + gamma*lam*last_gae*is_terminal

        target_val[t] = is_terminal * gamma * vpred[t+1] + roll["rew"][t]
    # target_val = gae_adv + roll["vpred"]

    return gae_adv, target_val
