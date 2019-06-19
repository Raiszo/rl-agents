import tensorflow as tf

class PPO_Agent:
    def __init__(self,
                 actor,
                 critic,
                 epsilon=0.2):
        self.actor = actor
        self.critic = critic
        self.epsilon = epsilon


    def action_value(self, obs):
        action, log_prob, logits = self.actor(obs)
        value = self.critic.predict(obs)

        return action , value, log_prob, logits


    def actor_loss(self, actions, advantages, obs, old_log_probs):
        _, new_log_probs, _, _, entropy = self.action_value(obs)
        ratio = tf.exp(new_log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + epsilon)
        
        surrogate_min = tf.minimum(ratio*advantages, clipped_ratio*advantages)
        surrogate_loss = tf.reduce_mean(surrogate_min)

        entropy_loss = tf.reduce_mean(entropy)
        
        return surrogate_loss + entropy_loss


    def critic_loss(self, values, target_values):
        value_loss = tf.losses.mean_squared_error(labels=target_values, predictions=values)
        value_loss = tf.reduce_mean(value_loss)

        return value_loss


    @tf.function
    def run_batch(self, optimizers, observations, target_values, advantages):
        
        with tf.GradientTape() as act_tape, tf.GradientTape() as crt_tape:
            actions, log_prob, dists, logits = self.actor(observations)
            values = self.critic(observations)

            act_loss = self.actor_loss(actions, advantages, dists, log_prob)
            crt_loss = self.critic_loss(values, target_values)

        gradients_of_actor = act_tape.gradient(act_loss, self.actor.trainable_variables)
        gradients_of_critic = crt_tape.gradient(crt_loss, self.critic.trainable_variables)

        optimizers[0].apply_gradients(zip(gradients_of_actor, self.actor.trainable_variables))
        optimizers[1].apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))


    def run_iteration(self, env,
                      obs, val, advs,
                      num_epochs=10, batch_size=64, learning_rate=1e-4):

        actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizers = [ actor_optimizer, critic_optimizer ]

        for _ in range(num_epochs):
            for i in range(int(ceil(size/batch_size))):
                start_idx = (i*batch_size)%size
                idx = train_indicies[start_idx:start_idx+batch_size]

                self.run_batch(optimizers, obs[idx, :], val[idx], advs[idx])

