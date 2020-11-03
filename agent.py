"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import os
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
               timeout_factor=3, fc_layer_params=(100,), learning_rate=1e-3,
               epsilon_greedy=0.1, penalty_per_step=0.0):
    self.num_actions = num_actions
    self.step_size = step_size
    self.timeout_factor = timeout_factor
    self.fc_layer_params = fc_layer_params
    self.learning_rate = learning_rate
    self.epsilon_greedy = epsilon_greedy
    self.penalty_per_step = penalty_per_step

    # variable for determining learning target mastery
    self.learning_score = 0

    # summary writer for tensorboard
    self.writer = tf.summary.create_file_writer('logs/')
    self.writer.set_as_default()

    # defining items which are tracked in checkpointer
    self.q_net, self.agent = self.build_agent()
    self.replay_buffer = self.build_replay_buffer()
    self.monster_speed = tf.Variable(initial_monster_speed, dtype=tf.float64)
    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()

    # defining other training items dependent on checkpointer parameters
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.driver, self.iterator = self.build_driver()

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

  def save_progress(self, step):
    """Save checkpoints and updated stats. Ignore keyboard interruptions."""
    def save_successfully():
      self.checkpointer.save(step)  # step is used as name
      tf.summary.scalar('monster_speed', self.monster_speed, step)
      tf.summary.scalar('learning_score', self.learning_score, step)
      for i, layer in enumerate(self.q_net.layers[0].get_weights()):
        tf.summary.histogram(f'layer{i}', layer, step)
      for i, layer in enumerate(self.q_net.layers[1].get_weights()):
        tf.summary.histogram(f'final_layer{i}', layer, step)
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

  def run_save(self, step):
    """Call save_progress and print out key statistics."""
    print('')
    self.save_progress(step)

    if step % video_interval == 0:
      episode_as_video(self.py_eval_env, self.agent.policy,
                       f'episode-{step}', self.tf_eval_env)

    # determine if agent should move on to next learning target
    # using a threshold of 80% for now
    if self.learning_score >= 0.8 * (num_eval := save_interval // eval_interval):
      print('Agent is very smart. Increasing the monster speed ...')
      self.monster_speed.assign_add(0.01)
      self.reset()

    print(f'Completed {step} training episodes.')
    print(f'Monster speed: {round(self.monster_speed.numpy().item(), 3)}.')
    print(f'Score over last {num_eval} evaluations: {self.learning_score}')
    self.learning_score = 0
    print('_' * 80)

  def run_eval(self, step):
    """Evaluate agent and print out key statistics."""
    # tracking some statistics every evaluation
    reward, n_steps = self.evaluate_agent()
    tf.summary.scalar('reward', reward, step)
    tf.summary.scalar('n_env_steps', n_steps, step)

    if reward >= 1.0:
      print(success_symbol, end='', flush=True)
      self.learning_score += 1
    else:
      print(fail_symbol, end='', flush=True)

  def train_ad_infinitum(self):
    """Train the agent until interrupted by user."""

    print_legend()
    while True:
      train_step = self.agent.train_step_counter.numpy().item()
      self.driver.run()  # run a single episode
      experience, _ = next(self.iterator)
      self.agent.train(experience)

      if train_step % save_interval == 0:
        self.run_save(train_step)

      if train_step % eval_interval == 0:
        self.run_eval(train_step)


# a few constant global variables
eval_interval = 10
save_interval = 200
video_interval = 2000
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
