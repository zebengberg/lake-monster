"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import os
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from environment import LakeMonsterEnvironment
from renderer import episode_as_video


# suppressing some warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Agent:
  """A class to hold global variables for tf_agent training."""
  # hyperparameters
  replay_buffer_max_length = 100000
  batch_size = 64

  def __init__(self, num_actions=4, step_size=0.1, initial_monster_speed=1.0,
               timeout_factor=3, fc_layer_params=100, learning_rate=1e-3,
               epsilon_greedy=0.1, penalty_per_step=0.0):
    self.num_actions = num_actions
    self.step_size = step_size
    self.timeout_factor = timeout_factor
    self.fc_layer_params = fc_layer_params
    self.learning_rate = learning_rate
    self.epsilon_greedy = epsilon_greedy
    self.penalty_per_step = penalty_per_step

    # summary writer for tensorboard
    self.summary_writer = tf.summary.create_file_writer('logs/')
    self.summary_writer.set_as_default()

    # defining items which are tracked in checkpointer
    self.q_net, self.agent = self.build_agent()
    self.replay_buffer = self.build_replay_buffer()
    self.monster_speed = tf.Variable(initial_monster_speed, dtype=tf.float64)
    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()

    # defining other training items dependent on checkpointer parameters
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.driver, self.iterator = self.build_driver()

    # self.stats = Stats()

    #self.weight_indices = self.build_weight_indices()

  def reset(self):
    """Reset member variables after updating monster_speed."""
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.driver, self.iterator = self.build_driver()

  def build_agent(self):
    """Build DQN agent with QNetwork."""
    py_temp_env = LakeMonsterEnvironment(num_actions=self.num_actions)
    tf_temp_env = tf_py_environment.TFPyEnvironment(py_temp_env)
    q_net = q_network.QNetwork(
        tf_temp_env.observation_spec(),
        tf_temp_env.action_spec(),
        fc_layer_params=self.fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    agent = dqn_agent.DqnAgent(
        tf_temp_env.time_step_spec(),
        tf_temp_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=self.epsilon_greedy,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0, dtype=tf.int64))

    agent.train = common.function(agent.train)
    return q_net, agent

  def build_replay_buffer(self):
    """Build replay buffer."""
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=self.agent.collect_data_spec,
        batch_size=1,
        max_length=self.replay_buffer_max_length)

  def build_checkpointer(self):
    """Build checkpointer."""
    return common.Checkpointer(
        ckpt_dir='checkpoints/',
        max_to_keep=3,
        # the rest of the parameters are optional kwargs
        agent=self.agent,
        policy=self.agent.policy,
        replay_buffer=self.replay_buffer,
        train_step_counter=self.agent.train_step_counter,
        monster_speed=self.monster_speed)

  def build_envs(self):
    """Build training and evaluation environments."""
    params = {'monster_speed': round(self.monster_speed.numpy().item(), 3),
              'timeout_factor': self.timeout_factor,
              'step_size': self.step_size,
              'num_actions': self.num_actions,
              'penalty_per_step': self.penalty_per_step}
    py_train_env = LakeMonsterEnvironment(**params)
    tf_train_env = tf_py_environment.TFPyEnvironment(py_train_env)
    py_eval_env = LakeMonsterEnvironment(**params)
    tf_eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
    return tf_train_env, py_eval_env, tf_eval_env

  def build_driver(self):
    """Build elements of the data pipeline."""
    observers = [self.replay_buffer.add_batch]
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env=self.tf_train_env,
        policy=self.agent.collect_policy,
        observers=observers)

    dataset = self.replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=self.batch_size,  # not sure!
        num_steps=2).prefetch(3)
    iterator = iter(dataset)
    return driver, iterator

  def confirm_hyperparameters(self):
    """Confirm that hyperparameters passed into self agree with tf objects."""
    print('Confirming hyperparameters ...')

    # NN parameters
    optimizer = self.agent._optimizer
    assert optimizer.get_config()['learning_rate'] == self.learning_rate

    # layers is list of all layers except for final output layer
    layers = self.q_net.layers[0].get_weights()
    for i, l in enumerate(layers):
      if i % 2 == 0:
        assert l.shape[1] == self.fc_layer_params[i // 2]
    final_layer = self.q_net.layers[1].get_weights()
    assert final_layer[0].shape[1] == self.num_actions

    # agent parameters
    assert self.agent._epsilon_greedy == self.epsilon_greedy

    # environment parameters
    n = self.tf_train_env.action_spec().maximum - self.tf_train_env.action_spec().minimum
    assert n + 1 == self.num_actions
    assert self.py_eval_env.step_size == self.step_size
    assert self.py_eval_env.timeout_factor == self.timeout_factor
    assert self.py_eval_env.penalty_per_step == self.penalty_per_step

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

  # def build_weight_indices(self, num_indices=20):
  #   """Select or load NN weight indices to be tracked over training."""
  #   indices = self.stats.get_weight_indices()
  #   if indices is not None:
  #     return indices

  #   # choosing indices for the first time from first layer of q_net
  #   x, y = self.q_net.layers[0].get_weights()[0].shape
  #   flat_indices = np.random.choice(x * y, size=num_indices, replace=False)
  #   return [divmod(i, y) for i in flat_indices]

  # def get_sample_weights(self):
  #   """Get current values of NN weights."""
  #   weights = self.q_net.layers[0].get_weights()[0]
  #   return {i: weights[i].item() for i in self.weight_indices}

  def save_progress(self):
    """Save checkpoints and updated stats. Ignore keyboard interruptions."""
    def save_successfully():
      self.checkpointer.save(self.agent.train_step_counter)  # arg used as name
      # self.stats.save()
      return True

    is_saved = False
    tried_to_interrupt = False
    print('Saving progress to disk ...')
    while not is_saved:
      try:
        is_saved = save_successfully()
      except KeyboardInterrupt:
        print('I will interrupt as soon as I am done saving!')
        tried_to_interrupt = True
        continue
    print('Progress saved.')
    if tried_to_interrupt:
      raise KeyboardInterrupt

  # def has_reached_mastery(self, score, num_episodes):
  #   """Determine if the agent has reached mastery of the learning target."""
  #   if score > 0.7:
  #     # upping the monster speed according to num_episodes
  #     self.monster_speed += 0.1 * 200 / num_episodes
  #     self.monster_speed = round(self.monster_speed, 3)
  #     print('Agent has mastered the learning target!')
  #     print(f'Increasing the monster speed to {self.monster_speed} ...')
  #     self.reset()

  #     # TODO: consider upping step_size, timeout_factor

  def train_ad_infinitum(self):
    """Train the agent until interrupted by user."""
    self.confirm_hyperparameters()
    print_legend()
    learning_score = 0

    while True:
      train_step = self.agent.train_step_counter.numpy().item()

      self.driver.run()  # run a single episode
      experience, _ = next(self.iterator)
      # tf.summary.trace_on()
      self.agent.train(experience)
      #tf.summary.trace_export(name='graph', step=train_step)

      if train_step % eval_interval == 0:
        reward, n_steps = self.evaluate_agent()
        # d = {'episode': train_step, 'reward': reward,
        #      'n_env_steps': n_steps, 'monster_speed': self.monster_speed,
        #      'loss': loss, 'weights': self.get_sample_weights()}
        # self.stats.add(d)

        if reward == 1.0:
          print(success_symbol, end='', flush=True)
          learning_score += 1
        else:
          print(fail_symbol, end='', flush=True)
          learning_score -= 1

      if train_step % save_interval == 0:
        tf.summary.scalar('reward', reward, train_step)
        tf.summary.scalar('n_env_steps', n_steps, train_step)
        tf.summary.scalar('monster_speed', self.monster_speed, train_step)
        for i, layer in enumerate(self.q_net.layers[0].get_weights()):
          tf.summary.histogram(f'layer{i}', layer, train_step)
        for i, layer in enumerate(self.q_net.layers[1].get_weights()):
          tf.summary.histogram(f'final_layer{i}', layer, train_step)

        print('')
        self.save_progress()
        print(f'Completed {train_step} episodes.')
        print(f'Monster speed: {round(self.monster_speed.numpy().item(), 3)}.')
        print(f'Learning score: {learning_score}')

        if learning_score == 10:
          self.monster_speed.assign_add(0.1)
          self.reset()
        learning_score = 0
        # avg_reward = self.stats.get_average_reward(self.monster_speed)
        # print(f'Over last eval period, the average reward is {avg_reward}.')
        # num_episodes = self.stats.get_number_episodes_on_lt(self.monster_speed)
        # print(f'The agent has trained {num_episodes} episodes for this LT.')
        # self.has_reached_mastery(avg_reward, num_episodes)

      if train_step % video_interval == 0:
        vid_file = f'episode-{train_step}'
        episode_as_video(self.py_eval_env, self.agent.policy,
                         vid_file, self.tf_eval_env)


eval_interval = 10
save_interval = 100
video_interval = 1000
success_symbol = '$'
fail_symbol = '|'


def print_legend():
  """Print command line training legend."""
  print('\n' + '#' * 80)
  print('          TRAINING LEGEND')
  print(success_symbol + ' = success on last evaluation episode')
  print(fail_symbol + ' = failure on last evaluation episode')
  print(f'Evaluation occurs every {eval_interval} episodes')
  print(f'Progress is saved every {save_interval} episodes')
  print(f'Videos are rendered every {video_interval} episodes')
  print('#' * 80 + '\n')
