import tensorflow as tf
from ppo.policy import Actor, Critic

class PPO_Agent(tf.keras.Model):
    def __init__(self,
                 actor,
                 critic,
                 epsilon=0.2):
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon

    def action_value(self, obs):
        action, log_prob, dist, logits = self.actor.predict(obs)
        value = self.critic.predict(obs)

        return action , value, log_prob, dist


    def actor_loss(self, actions, advantages, dist, log_probs):
        ratio = tf.exp(dist.log_prob(actions) - log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + epsilon)
        
        surrogate_min = tf.minimum(ratio*advantages, clipped_ratio*advantages)
        surrogate_loss = tf.reduce_mean(surrogate_min)

        entropy_loss = tf.reduce_mean(dist.entropy())
        
        return surrogate_loss + entropy_loss


    def critic_loss(self, values, target_values):
        value_loss = tf.losses.mean_squared_error(labels=target_values, predictions=values)
        value_loss = tf.reduce_mean(value_loss)

        return value_loss


    def compile(self, learning_rate):
        self.actor.compile(
            optimizer=tf.keras.optimizer.Adam(learning_rate),
            loss=self.actor_loss
        )

        self.critic.compile(
            optimizer=tf.keras.optimizer.Adam(learning_rate),
            loss=self.critic_loss
        )


    def run_iteration(self, env,
                      num_epochs=10, batch_size=64, learning_rate=1e-4):
        
        losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

        for _ in range(num_epochs):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]

                actor_loss = self.actor.train_on_batch()

                feed_dict = {
                    self.agent.state: obs[idx, :],
                    self.ac_na: acs[idx, :],
                    self.log_p: log_probs[idx, :],
                    self.adv_n: advs[idx],
                    self.t_val: val[idx],
                }

                stuff = sess.run(self.variables, feed_dict)

