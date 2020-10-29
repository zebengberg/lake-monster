"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import os
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment, wrappers
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from environment import LakeMonsterEnvironment
from renderer import episode_as_video
from stats import Stats

# suppressing some warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Agent:
  """A class to hold global variables for tf_agent training."""
  # hyperparameters
  replay_buffer_max_length = 100000
  batch_size = 64
  learning_rate = 1e-3
  fc_layer_params = (100,)

  monster_speed = 1.3
  num_actions = 20

  def __init__(self):
    self.stats = Stats()
    if s := self.stats.get_last_monster_speed():
      self.monster_speed = s
    else:
      self.monster_speed = 1.5

    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.agent = self.build_agent()
    self.replay_buffer, self.driver, self.iterator = self.build_data_pipeline()
    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()

  def build_envs(self):
    """Initialize environments based on monster_speed."""
    py_train_env = LakeMonsterEnvironment(self.monster_speed)
    discrete_py_train_env = wrappers.ActionDiscretizeWrapper(
        py_train_env, num_actions=self.num_actions)
    tf_train_env = tf_py_environment.TFPyEnvironment(discrete_py_train_env)
    py_eval_env = LakeMonsterEnvironment(self.monster_speed)
    discrete_py_eval_env = wrappers.ActionDiscretizeWrapper(
        py_eval_env, num_actions=self.num_actions)
    tf_eval_env = tf_py_environment.TFPyEnvironment(discrete_py_eval_env)
    return tf_train_env, py_eval_env, tf_eval_env

  def build_agent(self):
    """Build DQN agent with QNetwork."""
    q_net = q_network.QNetwork(
        self.tf_train_env.observation_spec(),
        self.tf_train_env.action_spec(),
        fc_layer_params=self.fc_layer_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    agent = dqn_agent.DqnAgent(
        self.tf_train_env.time_step_spec(),
        self.tf_train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0))

    agent.train = common.function(agent.train)
    return agent

  def build_data_pipeline(self):
    """Build replay buffer, driver, and train data iterator."""
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=self.agent.collect_data_spec,
        batch_size=self.tf_train_env.batch_size,  # defaults to 1
        max_length=self.replay_buffer_max_length)

    observers = [replay_buffer.add_batch]
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env=self.tf_train_env,
        policy=self.agent.collect_policy,
        observers=observers)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=self.batch_size,  # not sure!
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    return replay_buffer, driver, iterator

  def build_checkpointer(self):
    """Build checkpointer in order to save policy weights."""
    return common.Checkpointer(
        ckpt_dir='checkpoints/',
        max_to_keep=3,
        agent=self.agent,
        policy=self.agent.policy,
        replay_buffer=self.replay_buffer,
        global_step=self.agent.train_step_counter
    )

  def evaluate_agent(self):
    """Use naive while loop to evaluate policy in single episode."""
    n_steps = 0

    ts = self.tf_eval_env.reset()
    n_steps = 0
    while not ts.is_last():
      action = self.agent.policy.action(ts)
      ts = self.tf_eval_env.step(action.action)
      n_steps += 1

    reward = ts.reward.numpy()[0].item()  # item converts to native python type
    return reward, n_steps

  def save_progress(self):
    """Save train checkpoint and updated stats."""
    print('Saving progress to disk ...')
    self.checkpointer.save(self.agent.train_step_counter)
    self.stats.save()
    print('Progress saved.')

  def train_ad_infinitum(self):
    """Train the agent until interrupted by user."""

    while True:
      self.driver.run()  # train a single episode
      experience, _ = next(self.iterator)
      loss = self.agent.train(experience).loss.numpy().item()
      # print(loss)
      train_step = self.agent.train_step_counter.numpy().item()
      if train_step % 10 == 0:
        print('|', end='', flush=True)
        reward, n_steps = self.evaluate_agent()
        d = {'episode': train_step, 'reward': reward,
             'n_env_steps': n_steps, 'monster_speed': self.monster_speed,
             'loss': loss}
        self.stats.add(d)

      if train_step % 300 == 0:
        print('')
        self.save_progress()
        print('Progress saved!')
        vid_file = f'videos/episode-{train_step}.mp4'
        episode_as_video(self.py_eval_env, self.agent.policy,
                         vid_file, self.tf_eval_env)

        if self.stats.get_average_reward() > 0.8:
          # TODO: change monster_speed here
          print('WOW, super strong!')


if __name__ == '__main__':
  a = Agent()
  a.train_ad_infinitum()

  # print(q_net.layers[0].get_weights()[0].shape)
  # print(q_net.layers[0].get_weights()[1].shape)
  # print(q_net.layers[1].get_weights()[0].shape)
  # print(q_net.layers[1].get_weights()[1].shape)
