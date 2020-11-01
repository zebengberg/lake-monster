"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import os
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment, wrappers
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
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

  def __init__(self, num_actions=4, step_size=0.1, initial_monster_speed=1.0,
               timeout_factor=3, hidden_layer_nodes=100):
    self.num_actions = num_actions
    self.step_size = step_size
    self.timeout_factor = timeout_factor
    self.fc_layer_params = (hidden_layer_nodes,)

    self.stats = Stats()
    self.monster_speed = self.stats.get_last_monster_speed()
    if self.monster_speed is None:
      self.monster_speed = initial_monster_speed

    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.q_net, self.agent = self.build_agent()
    self.replay_buffer, self.driver, self.iterator = self.build_data_pipeline()
    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()
    self.weight_indices = self.build_weight_indices()

  def reset(self):
    """Reset variables after updating monster speed."""
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.q_net, self.agent = self.build_agent()
    self.replay_buffer, self.driver, self.iterator = self.build_data_pipeline()
    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()

  def build_envs(self):
    """Initialize environments based on monster_speed."""
    params = {'monster_speed': self.monster_speed,
              'timeout_factor': self.timeout_factor,
              'step_size': self.step_size}
    py_train_env = LakeMonsterEnvironment(**params)
    discrete_py_train_env = wrappers.ActionDiscretizeWrapper(
        py_train_env, num_actions=self.num_actions)
    tf_train_env = tf_py_environment.TFPyEnvironment(discrete_py_train_env)
    py_eval_env = LakeMonsterEnvironment(**params)
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
    return q_net, agent

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

    proportion_steps = n_steps / self.py_eval_env.duration
    reward = ts.reward.numpy()[0].item()  # item converts to native python type
    return reward, proportion_steps

  def build_weight_indices(self, num_indices=20):
    """Select or load NN weight indices to be tracked over training."""
    indices = self.stats.get_weight_indices()
    if indices is not None:
      return indices

    # choosing indices for the first time from first layer of q_net
    x, y = self.q_net.layers[0].get_weights()[0].shape
    flat_indices = np.random.choice(x * y, size=num_indices, replace=False)
    return [divmod(i, y) for i in flat_indices]

  def get_sample_weights(self):
    """Get current values of NN weights."""
    weights = self.q_net.layers[0].get_weights()[0]
    return {i: weights[i].item() for i in self.weight_indices}

  def save_progress(self):
    """Save checkpoints and updated stats. Ignore keyboard interruptions."""
    def save_successfully():
      self.checkpointer.save(self.agent.train_step_counter)
      self.stats.save()
      return True

    is_saved = False
    tried_to_interrupt = False
    print('Saving progress to disk ...')
    while not is_saved:
      try:
        is_saved = save_successfully()
      except KeyboardInterrupt:
        tried_to_interrupt = True
        continue
    print('Progress saved.')
    if tried_to_interrupt:
      raise KeyboardInterrupt

  def train_ad_infinitum(self):
    """Train the agent until interrupted by user."""
    print_legend()
    while True:
      self.driver.run()  # train a single episode
      experience, _ = next(self.iterator)
      loss = self.agent.train(experience).loss.numpy().item()
      train_step = self.agent.train_step_counter.numpy().item()

      if train_step % eval_interval == 0:
        reward, n_steps = self.evaluate_agent()
        d = {'episode': train_step, 'reward': reward,
             'n_env_steps': n_steps, 'monster_speed': self.monster_speed,
             'loss': loss, 'weights': self.get_sample_weights()}
        self.stats.add(d)
        if reward == 1.0:
          print('$', end='', flush=True)
        else:
          print('.', end='', flush=True)

      if train_step % save_interval == 0:
        print(f'\nCompleted {train_step} episodes.')
        print(f'Current monster speed is {self.monster_speed}.')
        self.save_progress()

        if (a := self.stats.get_average_reward(self.monster_speed)) > 0.4:
          # upping the monster speed in proportion to performance
          self.monster_speed += 0.1 * a
          self.monster_speed = round(self.monster_speed, 3)
          print('Agent is very strong!')
          print(f'Increasing the monster speed to {self.monster_speed} ...')
          self.reset()

      if train_step % video_interval == 0:
        vid_file = f'episode-{train_step}.mp4'
        episode_as_video(self.py_eval_env, self.agent.policy,
                         vid_file, self.tf_eval_env)


eval_interval = 10
save_interval = 200
video_interval = 2000


def print_legend():
  """Print command line training legend."""
  print('\n' + '#' * 80)
  print('          TRAINING LEGEND')
  print('$ = success on last evaluation episode')
  print('. = failure on last evaluation episode')
  print(f'Evaluation occurs every {eval_interval} episodes')
  print(f'Progress is saved every {save_interval} episodes')
  print(f'Videos are rendered and saved every {video_interval} episodes')
  print('#' * 80 + '\n')
