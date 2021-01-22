"""Log tf graph tensorboard and verify agent."""

import os
import tensorflow as tf
from lake_monster.agent.agent import Agent
from lake_monster import configs


class ModuleWrapper(tf.Module):
  """Wrap a tf-agent QNetwork as a tf.Module in order to log tf graph."""

  def __init__(self, net, use_categorical):
    super().__init__()
    self.net = net
    self.use_categorical = use_categorical

  @tf.function
  def __call__(self, x):
    if self.use_categorical:
      assert len(self.net.layers) == 1
      layers = self.net.layers[0].layers
    else:
      assert len(self.net.layers) == 2
      layers = self.net.layers

    layers = layers[0].layers + [layers[1]]
    for layer in layers:
      x = layer(x)
    return x


class ModelWrapper(tf.keras.Model):
  """Wrap a tf-agent QNetwork as a tf.Model."""

  def __init__(self, net, use_categorical):
    super().__init__()
    self.net = net
    self.use_categorical = use_categorical

  def call(self, x):  # pylint: disable=arguments-differ
    if self.use_categorical:
      assert len(self.net.layers) == 1
      layers = self.net.layers[0].layers
    else:
      assert len(self.net.layers) == 2
      layers = self.net.layers

    layers = layers[0].layers + [layers[1]]
    for layer in layers:
      x = layer(x)
    return x


def log_graph(uid, params=None, write_logs=True):
  """Call tf.summary.trace"""
  if params is None:
    params = {}
  a = Agent(uid, **params)

  model = ModuleWrapper(a.q_net.copy(), a.use_categorical)

  x = a.tf_env.reset().observation  # input to model.__call__
  if write_logs:
    logdir = os.path.join(configs.LOG_DIR, uid)
    summary_writer = tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()
    tf.summary.trace_on()
    model(x)  # ignoring output
    tf.summary.trace_export(name='q_net', step=0)
  else:
    model(x)  # ignoring output


def verify_agent(uid, params=None):
  """Print network summary, confirm parameters, and run driver."""
  if params is None:
    params = {}
  a = Agent(uid, **params)
  print('Summary of underlying neural network:')
  a.q_net.summary()

  assert a.agent._optimizer.get_config()['learning_rate'] == a.learning_rate

  # neural network layer shapes
  weights = [w for layers in a.q_net.layers for w in layers.get_weights()]
  for arr, size in zip(weights[::2], a.fc_layer_params):
    assert arr.shape[1] == size
  if a.use_categorical:
    assert weights[-2].shape[1] == a.n_actions * a.q_net.num_atoms
  else:
    assert weights[-2].shape[1] == a.n_actions

  # agent parameters
  assert a.agent._epsilon_greedy == a.epsilon_greedy
  assert a.agent.train_sequence_length == a.n_step_update + 1

  # environment parameters
  n = a.tf_env.action_spec().maximum - a.tf_env.action_spec().minimum
  assert a.n_actions == n + 1
  assert a.monster_speed.numpy().item() == a.py_env.monster_speed
  assert a.timeout_factor == a.py_env.timeout_factor
  assert a.n_actions == a.py_env.n_actions
  assert a.step_size.numpy().item() == a.py_env.step_size
  assert a.use_mini_rewards == a.py_env.use_mini_rewards

  for _ in range(10):
    a.driver.run()
