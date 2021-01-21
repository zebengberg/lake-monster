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
from lake_monster.environment.environment import LakeMonsterEnvironment
from lake_monster.environment.variations import MultiMonsterEnvironment, JumpingEnvironment
from lake_monster.environment.animate import episode_as_video
from lake_monster.agent.evaluate import evaluate_episode, probe_policy
from lake_monster.utils import log_results, py_to_tf
from lake_monster import configs


# suppressing some annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


FORMATIVE_INTERVAL = 20
SAVE_INTERVAL = 200
SUMMATIVE_INTERVAL = 2000
NUM_EVALS = SAVE_INTERVAL // FORMATIVE_INTERVAL
SUCCESS_SYMBOL = '$'
FAIL_SYMBOL = '|'


def print_legend():
  """Print command line training legend."""
  print('\n' + '#' * 65)
  print('          TRAINING LEGEND')
  print(SUCCESS_SYMBOL + ' = success on last formative evaluation')
  print(FAIL_SYMBOL + ' = failure on last formative evaluation')
  print('')
  print(f'Formative evaluation occurs every {FORMATIVE_INTERVAL} episodes.')
  print('This includes logging training metrics for TensorBoard.')
  print('')
  print(f'Progress is saved every {SAVE_INTERVAL} episodes.')
  print('This includes checking mastery and writing checkpoints.')
  print('')
  print(f'Summative evaluation occurs every {SUMMATIVE_INTERVAL} episodes.')
  print('This includes saving policies, rendering videos, and logging results.')
  print('#' * 65 + '\n')


class Agent:
  """A class to hold global variables for tf_agent training."""
  replay_buffer_max_length = 1_000_000
  batch_size = 64

  def __init__(
          self,
          uid,
          n_actions=16,
          initial_step_size=0.1,
          initial_monster_speed=4.0,
          timeout_factor=2.0,
          use_mini_rewards=True,
          fc_layer_params=(100, 100),
          dropout_layer_params=None,
          learning_rate=0.001,
          epsilon_greedy=0.1,
          n_step_update=6,
          use_categorical=True,
          use_step_schedule=True,
          use_mastery=True,
          summative_callback=None):

    self.n_actions = n_actions
    self.initial_step_size = initial_step_size
    self.initial_monster_speed = initial_monster_speed
    self.timeout_factor = timeout_factor
    self.use_mini_rewards = use_mini_rewards
    self.fc_layer_params = fc_layer_params
    self.dropout_layer_params = dropout_layer_params
    self.learning_rate = learning_rate
    self.epsilon_greedy = epsilon_greedy
    self.n_step_update = n_step_update
    self.use_categorical = use_categorical
    self.use_step_schedule = use_step_schedule
    self.use_mastery = use_mastery
    self.summative_callback = summative_callback

    # variable for determining learning target mastery
    self.learning_score = 0
    self.reward_sum = 0

    # summary writer for tensorboard
    self.uid = tf.Variable(uid, dtype=tf.string)
    log_dir = os.path.join(configs.LOG_DIR, self.get_uid())
    self.writer = tf.summary.create_file_writer(log_dir)
    self.writer.set_as_default()

    # defining items which are tracked in checkpointer
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
    self.py_env, self.tf_env = self.build_env()
    self.driver, self.iterator = self.build_driver()

  def get_uid(self):
    """Return UID as string."""
    return self.uid.numpy().decode()

  @property
  def env_params(self):
    """Return dictionary of environment parameters."""
    return {'monster_speed': self.monster_speed.numpy().item(),
            'timeout_factor': self.timeout_factor,
            'step_size': self.step_size.numpy().item(),
            'n_actions': self.n_actions,
            'use_mini_rewards': self.use_mini_rewards}

  def reset(self):
    """Reset member variables after updating monster_speed."""
    self.py_env, self.tf_env = self.build_env()
    self.driver, self.iterator = self.build_driver()

  def build_dqn_agent(self):
    """Build DQN agent with QNetwork."""
    temp_env = self.build_temp_env()

    q_net = q_network.QNetwork(
        temp_env.observation_spec(),
        temp_env.action_spec(),
        fc_layer_params=self.fc_layer_params,
        dropout_layer_params=self.dropout_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    agent = DqnAgent(
        temp_env.time_step_spec(),
        temp_env.action_spec(),
        n_step_update=self.n_step_update,
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=self.epsilon_greedy,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0, dtype=tf.int64))

    return q_net, agent

  def build_categorical_dqn_agent(self):
    """Build categorical DQN agent with CategoricalQNetwork."""
    temp_env = self.build_temp_env()

    if self.dropout_layer_params is not None:
      raise AttributeError('CategoricalQNetwork does accept dropout layers.')

    q_net = categorical_q_network.CategoricalQNetwork(
        temp_env.observation_spec(),
        temp_env.action_spec(),
        fc_layer_params=self.fc_layer_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    agent = CategoricalDqnAgent(
        temp_env.time_step_spec(),
        temp_env.action_spec(),
        n_step_update=self.n_step_update,
        categorical_q_network=q_net,
        optimizer=optimizer,
        min_q_value=0.0,
        max_q_value=3.0,
        epsilon_greedy=self.epsilon_greedy,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0, dtype=tf.int64))

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
        ckpt_dir=configs.CHECKPOINT_DIR,
        max_to_keep=3,
        # the rest of the parameters are optional kwargs
        agent=self.agent,
        policy=self.agent.policy,
        replay_buffer=self.replay_buffer,
        train_step_counter=self.agent.train_step_counter,
        uid=self.uid,
        monster_speed=self.monster_speed,
        step_size=self.step_size)

  def build_temp_env(self):
    """Helper function for build_dqn_agent."""
    temp_env = LakeMonsterEnvironment(n_actions=self.n_actions)
    return TFPyEnvironment(temp_env)

  def build_env(self):
    """Build training and evaluation environments."""
    params = self.env_params
    py_env = LakeMonsterEnvironment(**params)
    tf_env = TFPyEnvironment(py_env)
    return py_env, tf_env

  def build_driver(self):
    """Build elements of the data pipeline."""
    observers = [self.replay_buffer.add_batch]
    driver = DynamicEpisodeDriver(
        env=self.tf_env,
        policy=self.agent.collect_policy,
        observers=observers)

    dataset = self.replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=self.batch_size,
        num_steps=self.agent.train_sequence_length
    ).prefetch(3)
    iterator = iter(dataset)
    return driver, iterator

  def save_checkpoint_and_logs(self, step):
    """Save checkpoint and write metrics and weights with tf.summary."""
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

  def save_policy(self, step):
    """Save strong policy with tf-agent PolicySaver."""
    print('Saving agent policy.')

    # saving environment params as metadata in order to reconstruct environment
    metadata = py_to_tf(self.env_params)
    saver = PolicySaver(self.agent.policy,
                        train_step=self.agent.train_step_counter,
                        metadata=metadata,
                        batch_size=None)
    dir_name = f'{self.uid.numpy().decode()}-{step}'
    filepath = os.path.join(configs.POLICY_DIR, dir_name)
    saver.save(filepath)
    print('Policy saved.')

  def run_save(self, step):
    """Check for mastery and write checkpoints."""
    self.save_checkpoint_and_logs(step)
    if self.use_mastery:
      self.check_mastery()
    if self.use_step_schedule:
      self.check_step_schedule(step)

    print(f'Completed {step} training episodes.')
    print(f'Monster speed: {self.monster_speed.numpy().item():.3f}.')
    print(f'Step size: {self.step_size.numpy().item():.2f}.')
    print(f'Formative passes: {self.learning_score} / {NUM_EVALS}')
    print(f'Avg reward on last formatives: {self.reward_sum / NUM_EVALS:.3f}')
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
    if step % 100_000 == 0:
      print('Decreasing the step size according to the schedule.')
      self.step_size.assign(tf.multiply(0.7, self.step_size))
      self.reset()

  def run_formative(self, step):
    """Evaluate agent once, print results, and log metrics for TensorBoard."""
    reward, n_steps = evaluate_episode(self.agent.policy, self.env_params)
    self.reward_sum += reward
    tf.summary.scalar('reward', reward, step)
    tf.summary.scalar('n_env_steps', n_steps, step)

    if reward >= 1.0:
      print(SUCCESS_SYMBOL, end='', flush=True)
      self.learning_score += 1
    else:
      print(FAIL_SYMBOL, end='', flush=True)
    if step % SAVE_INTERVAL == 0:
      print('')

  def run_summative(self, step):
    """Render a video of the agent, save a policy, and log results."""
    print('Creating video ...')
    filepath = os.path.join(configs.VIDEO_DIR, f'episode-{step}.mp4')
    episode_as_video(self.py_env, self.agent.policy, filepath)
    print('Evaluating agent. You will see several lines of dots ...')
    result = probe_policy(self.agent.policy, self.env_params)
    result['n_episode'] = step
    print('Logging evaluation results ...')
    print(result)
    log_results(self.get_uid(), result)
    self.save_policy(step)
    if self.summative_callback is None:
      return False
    return self.summative_callback()

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
      if train_step % FORMATIVE_INTERVAL == 0:
        self.run_formative(train_step)
      if train_step % SUMMATIVE_INTERVAL == 0:
        if self.run_summative(train_step):
          break
      if train_step % SAVE_INTERVAL == 0:
        self.run_save(train_step)


class MultiMonsterAgent(Agent):
  """A DQN agent for the MultiMonsterEnvironment."""

  def __init__(self, uid, n_monsters, **kwargs):
    self.n_monsters = n_monsters
    super().__init__(uid, **kwargs)

  @ property
  def env_params(self):
    """Override from Agent."""
    params = super().env_params
    params['n_monsters'] = self.n_monsters
    return params

  def build_temp_env(self):
    """Override from Agent."""
    temp_env = MultiMonsterEnvironment(n_actions=self.n_actions,
                                       n_monsters=self.n_monsters)
    return TFPyEnvironment(temp_env)

  def build_env(self):
    """Override from Agent."""
    params = self.env_params
    py_env = MultiMonsterEnvironment(**params)
    tf_env = TFPyEnvironment(py_env)
    return py_env, tf_env


class JumpingAgent(Agent):
  """A DQN agent for the JumpingEnvironment."""

  def __init__(self, uid, **kwargs):
    super().__init__(uid, **kwargs)

  @ property
  def env_params(self):
    """Override from Agent."""
    params = super().env_params
    params['is_jumping'] = True
    return params

  def build_temp_env(self):
    """Override from Agent."""
    temp_env = JumpingEnvironment(n_actions=self.n_actions)
    return TFPyEnvironment(temp_env)

  def build_env(self):
    """Override from Agent."""
    params = self.env_params
    py_env = JumpingEnvironment(**params)
    tf_env = TFPyEnvironment(py_env)
    return py_env, tf_env
