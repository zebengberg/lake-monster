"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""


import os
import warnings
import absl.logging
import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import CategoricalDqnAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.utils import common
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.policies.policy_saver import PolicySaver
from environment import LakeMonsterEnvironment
from renderer import episode_as_video
from utils import log_results, build_df_from_tf_logs


# suppressing some annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# a few constant global variables
EVAL_INTERVAL = 10
SAVE_INTERVAL = 100
VIDEO_INTERVAL = 1000
POLICY_INTERVAL = 1000
NUM_EVALS = SAVE_INTERVAL // EVAL_INTERVAL
SUCCESS_SYMBOL = '$'
FAIL_SYMBOL = '|'


def print_legend():
  """Print command line training legend."""
  print('\n' + '#' * 65)
  print('          TRAINING LEGEND')
  print(SUCCESS_SYMBOL + ' = success on last evaluation episode')
  print(FAIL_SYMBOL + ' = failure on last evaluation episode')
  print(f'Evaluation occurs every {EVAL_INTERVAL} episodes')
  print(f'Progress is saved every {SAVE_INTERVAL} episodes')
  print(f'Videos are rendered every {VIDEO_INTERVAL} episodes')
  print('#' * 65 + '\n')


class Agent:
  """A class to hold global variables for tf_agent training."""
  replay_buffer_max_length = 1_000_000
  batch_size = 64

  def __init__(
          self,
          uid,
          num_actions=8,
          initial_step_size=0.1,
          initial_monster_speed=3.3,
          timeout_factor=3.0,
          use_mini_rewards=True,
          use_cartesian=False,
          use_noisy_start=False,
          fc_layer_params=(50, 50),
          dropout_layer_params=None,
          learning_rate=0.0005,
          epsilon_greedy=0.1,
          n_step_update=10,
          use_categorical=False,
          use_step_schedule=False,
          use_learning_rate_schedule=False,
          use_mastery=True):

    self.num_actions = num_actions
    self.initial_step_size = initial_step_size
    self.initial_monster_speed = initial_monster_speed
    self.timeout_factor = timeout_factor
    self.use_mini_rewards = use_mini_rewards
    self.use_cartesian = use_cartesian
    self.use_noisy_start = use_noisy_start
    self.fc_layer_params = fc_layer_params
    self.dropout_layer_params = dropout_layer_params
    self.learning_rate = learning_rate
    self.epsilon_greedy = epsilon_greedy
    self.n_step_update = n_step_update
    self.use_categorical = use_categorical
    self.use_step_schedule = use_step_schedule
    self.use_learning_rate_schedule = use_learning_rate_schedule
    self.use_mastery = use_mastery

    # variable for determining learning target mastery
    self.learning_score = 0
    self.reward_sum = 0

    # summary writer for tensorboard
    self.writer = tf.summary.create_file_writer('logs/')
    self.writer.set_as_default()

    # defining items which are tracked in checkpointer
    self.uid = tf.Variable(uid, dtype=tf.string)
    self.monster_speed = tf.Variable(initial_monster_speed, dtype=tf.float64)
    self.step_size = tf.Variable(initial_step_size, dtype=tf.float64)
    if self.use_categorical:
      self.dropout_layer_params = None  # overwriting
      self.q_net, self.agent = self.build_categorical_dqn_agent()
    else:
      self.q_net, self.agent = self.build_dqn_agent()
    self.replay_buffer = self.build_replay_buffer()

    self.checkpointer = self.build_checkpointer()
    self.checkpointer.initialize_or_restore()

    # defining other training items dependent on checkpointer parameters
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.driver, self.iterator = self.build_driver()

  def reset(self):
    """Reset member variables after updating monster_speed."""
    self.tf_train_env, self.py_eval_env, self.tf_eval_env = self.build_envs()
    self.driver, self.iterator = self.build_driver()

  def build_dqn_agent(self):
    """Build DQN agent with QNetwork."""
    py_temp_env = LakeMonsterEnvironment(num_actions=self.num_actions,
                                         use_cartesian=self.use_cartesian)
    tf_temp_env = TFPyEnvironment(py_temp_env)

    q_net = q_network.QNetwork(
        tf_temp_env.observation_spec(),
        tf_temp_env.action_spec(),
        fc_layer_params=self.fc_layer_params,
        dropout_layer_params=self.dropout_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    agent = DqnAgent(
        tf_temp_env.time_step_spec(),
        tf_temp_env.action_spec(),
        n_step_update=self.n_step_update,
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=self.epsilon_greedy,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0, dtype=tf.int64))

    return q_net, agent

  def build_categorical_dqn_agent(self):
    """Build categorical DQN agent with CategoricalQNetwork."""
    py_temp_env = LakeMonsterEnvironment(num_actions=self.num_actions,
                                         use_cartesian=self.use_cartesian)
    tf_temp_env = TFPyEnvironment(py_temp_env)

    if self.dropout_layer_params is not None:
      raise AttributeError('CategoricalQNetwork does accept dropout layers.')

    q_net = categorical_q_network.CategoricalQNetwork(
        tf_temp_env.observation_spec(),
        tf_temp_env.action_spec(),
        fc_layer_params=self.fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    agent = CategoricalDqnAgent(
        tf_temp_env.time_step_spec(),
        tf_temp_env.action_spec(),
        n_step_update=self.n_step_update,
        categorical_q_network=q_net,
        optimizer=optimizer,
        min_q_value=0.0,
        max_q_value=3.0,
        epsilon_greedy=self.epsilon_greedy,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0, dtype=tf.int64))
    agent.train = common.function(agent.train)

    return q_net, agent

  def build_replay_buffer(self):
    """Build replay buffer."""
    return TFUniformReplayBuffer(
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
        uid=self.uid,
        monster_speed=self.monster_speed,
        step_size=self.step_size)

  def build_envs(self):
    """Build training and evaluation environments."""
    params = {'monster_speed': self.monster_speed.numpy().item(),
              'timeout_factor': self.timeout_factor,
              'step_size': self.step_size.numpy().item(),
              'num_actions': self.num_actions,
              'use_mini_rewards': self.use_mini_rewards,
              'use_cartesian': self.use_cartesian,
              'use_noisy_start': self.use_noisy_start}
    py_train_env = LakeMonsterEnvironment(**params)
    tf_train_env = TFPyEnvironment(py_train_env)

    params['use_noisy_start'] = False  # don't want noisy start for evaluation
    py_eval_env = LakeMonsterEnvironment(**params)
    tf_eval_env = TFPyEnvironment(py_eval_env)

    return tf_train_env, py_eval_env, tf_eval_env

  def build_driver(self):
    """Build elements of the data pipeline."""
    observers = [self.replay_buffer.add_batch]
    driver = DynamicEpisodeDriver(
        env=self.tf_train_env,
        policy=self.agent.collect_policy,
        observers=observers)

    dataset = self.replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=self.batch_size,
        num_steps=self.agent.train_sequence_length
    ).prefetch(3)
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
    """Save checkpoints and write tf.summary. Ignore keyboard interruptions."""
    def save_successfully():
      self.checkpointer.save(step)  # step is used as name
      tf.summary.scalar('monster_speed', self.monster_speed, step)
      tf.summary.scalar('step_size', self.step_size, step)
      tf.summary.scalar('learning_score', self.learning_score, step)
      tf.summary.scalar('reward_sum', self.reward_sum, step)

      # self.q_net.layers either has 1 or 2 elements depending on whether q_net
      # is categorical or not
      weights = [w for layers in self.q_net.layers for w in layers.get_weights()]
      for i, w in enumerate(weights):
        tf.summary.histogram(f'layer{i}', w, step)
      return True

    is_saved = False
    tried_to_interrupt = False
    print('Saving checkpointer and logs to disk ...')
    while not is_saved:
      try:
        is_saved = save_successfully()
      except KeyboardInterrupt:
        print('I will interrupt as soon as I am done saving!')
        tried_to_interrupt = True
        continue
    print('Save successful.')
    if tried_to_interrupt:
      raise KeyboardInterrupt

  def save_policy(self, step):
    """Save strong policy with tf-agent PolicySaver."""
    print('Saving a strong policy.')
    # saving environment params as metadata in order to reconstruct environment

    metadata = {'monster_speed': self.monster_speed,  # already tf.Variable
                'step_size': self.step_size,  # already tf.Variable
                'timeout_factor': tf.Variable(self.timeout_factor),
                'num_actions': tf.Variable(self.num_actions)}
    saver = PolicySaver(self.agent.policy,
                        train_step=self.agent.train_step_counter,
                        metadata=metadata,
                        batch_size=None)
    dir_name = f'{self.uid.numpy().decode()}-{step}'
    filepath = os.path.join('policies', dir_name)
    saver.save(filepath)
    print('Policy saved.')

  def run_save(self, step):
    """Call save_progress and print out key statistics."""

    self.save_progress(step)
    if self.use_mastery:
      self.check_mastery()
    if self.use_step_schedule:
      self.check_step_schedule(step)

    print(f'Completed {step} training episodes.')
    print(f'Monster speed: {self.monster_speed.numpy().item():.3f}.')
    print(f'Step size: {self.step_size.numpy().item():.2f}.')
    print(f'Score over eval period: {self.learning_score} / {NUM_EVALS}')
    print(f'Avg reward over eval period: {self.reward_sum / NUM_EVALS:.3f}')
    self.learning_score = 0
    self.reward_sum = 0
    print('_' * 65)

  def check_mastery(self):
    """Determine if policy is sufficiently strong to increase monster_speed."""
    if self.learning_score >= 0.9 * NUM_EVALS:  # threshold 90%
      print('Agent is very smart. Increasing monster speed ...')
      if tf.math.greater(self.monster_speed, 3.4):
        self.monster_speed.assign_add(0.01)
      elif tf.math.greater(self.monster_speed, 3.0):
        self.monster_speed.assign_add(0.02)
      else:
        self.monster_speed.assign_add(0.04)
      self.reset()

  def check_step_schedule(self, step):
    """Determine if step_size should be decreased."""
    if step == 100_000 or step == 200_000:
      print('Decreasing the step size according to the schedule.')
      self.step_size.assign(tf.multiply(0.5, self.step_size))
      self.reset()

  def run_eval(self, step):
    """Evaluate agent and print out key statistics."""
    # tracking some statistics every evaluation
    reward, n_steps = self.evaluate_agent()
    self.reward_sum += reward
    tf.summary.scalar('reward', reward, step)
    tf.summary.scalar('n_env_steps', n_steps, step)

    if reward >= 1.0:
      print(SUCCESS_SYMBOL, end='', flush=True)
      self.learning_score += 1
    else:
      print(FAIL_SYMBOL, end='', flush=True)
    if (step + EVAL_INTERVAL) % SAVE_INTERVAL == 0:
      print('')

  def run_video(self, step):
    """Save a video and log results."""
    episode_as_video(self.py_eval_env, self.agent.policy,
                     f'episode-{step}', self.tf_eval_env)
    df = build_df_from_tf_logs()
    log_results(self.uid.numpy().decode(), df.to_dict(orient='list'))

  def train_ad_infinitum(self):
    """Train the agent until interrupted by user."""
    # some basic setup needed
    self.agent.train = common.function(self.agent.train)
    self.driver.run()
    self.driver.run()
    self.agent.initialize()
    print_legend()

    while True:
      self.driver.run()
      experience, _ = next(self.iterator)
      self.agent.train(experience)

      train_step = self.agent.train_step_counter.numpy().item()
      if train_step % POLICY_INTERVAL == 0:
        self.save_policy(train_step)
      if train_step % VIDEO_INTERVAL == 0:
        self.run_video(train_step)
      if train_step % SAVE_INTERVAL == 0:
        self.run_save(train_step)
      if train_step % EVAL_INTERVAL == 0:
        self.run_eval(train_step)
