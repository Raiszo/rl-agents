import datetime
from os import path

from rl_agents.env_utils import rollouts_generator, get_adv_vtarg

class Sensei:
    def __init__(self, agent, alg_name, env_fn,
                 ite=200, horizon=2048, epochs=20,
                 gamma=0.99, gae_lambda=0.95,
                 log_dir='logs'):
        self.agent = agent
        self.alg_name = alg_name
        self.env_fn = env_fn

        self.num_ite = ite
        self.horizon = horizon
        self.epochs = epochs
        self.gamma = 0.99
        self.gae_lambda = gae_lambda

        self.log_dir = log_dir

        is_continuous = self.agent.is_continuous
        self.generator = rollouts_generator(agent, env_fn(), is_continuous, horizon)

    def train(self):
        env = self.env_fn()

        folder_name = '{}_{}'.format(env.unwrapped.spec.id, alg_name)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = path.join(self.log_dir, folder_name, current_time)
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        for i in range(self.num_ite):
            rollout = self.generator.__next__()
            adv, target_value = get_adv_vtarg(rollout, lam=lam, gamma=gamma)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
            self.agent.run_ite(rollout['ob'], rollout['ac'], rollout['log_probs'], target_value, adv, epochs=epochs, batch_size=256)
            with summary_writer.as_default():
                tf.summary.scalar('reward mean', np.array(rollout["ep_rets"]).mean(), step=i)
    
            if i % 50 == 0 or i == num_ite-1:
                self.actor.save_weights(log_dir+'/_actor_'+str(i), save_format='tf')


    def test(self):
        pass
    


alg_name = "PPO"
num_ite = 200
lam = 0.95
gamma = 0.99
epochs = 10

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/{}_{}/'.format(env.unwrapped.spec.id, alg_name) + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for i in range(num_ite):
    rollout = generator_ppo.__next__()
    adv, target_value = get_adv_vtarg(rollout, lam=lam, gamma=gamma)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    jen_ppo.run_ite(rollout['ob'], rollout['ac'], rollout['log_probs'], target_value, adv, epochs=epochs, batch_size=256)
    with train_summary_writer.as_default():
        tf.summary.scalar('reward mean', np.array(rollout["ep_rets"]).mean(), step=i)
    
    if i % 50 == 0 or i == num_ite-1:
        actor_ppo.save_weights(train_log_dir+'/_actor_'+str(i), save_format='tf')
        critic_ppo.save_weights(train_log_dir+'/_critic_'+str(i), save_format='tf')
