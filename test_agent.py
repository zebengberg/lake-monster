"""A module for testing the Agent class and logging tf graph information."""


import tensorflow as tf
from agent import Agent


class ModelWrapper(tf.Module):
  """Wrap a tf-agent Q-network as a tf.Module in order to log tf graph."""

  def __init__(self, net):
    super().__init__()
    self.net = net

  @tf.function
  def __call__(self, x):
    for layer in self.net.layers[0].layers:
      x = layer(x)
    return self.net.layers[1](x)


def log_graph(params=None):
  """Call tf.summary.trace"""
  if params is None:
    params = {}
  a = Agent(**params)
  model = ModelWrapper(a.q_net.copy())

  summary_writer = tf.summary.create_file_writer('logs/')
  summary_writer.set_as_default()

  x = a.tf_eval_env.reset().observation  # input to model.__call__
  tf.summary.trace_on()
  model(x)
  tf.summary.trace_export(name='q_net', step=0)


def test_agent(params=None):
  """Print summary and confirm parameters passed into self agree with tf objects."""
  if params is None:
    params = {}
  a = Agent(**params)
  print('Summary of underlying neural network:')
  a.q_net.summary()

  assert a.agent._optimizer.get_config()['learning_rate'] == a.learning_rate

  # neural network layer shapes
  weights = [w for layers in a.q_net.layers for w in layers.get_weights()]
  for arr, size in zip(weights[::2], a.fc_layer_params):
    assert arr.shape[1] == size
  assert weights[-2].shape[1] == a.num_actions * a.q_net.num_atoms

  # agent parameters
  assert a.agent._epsilon_greedy == a.epsilon_greedy

  # environment parameters
  n = a.tf_train_env.action_spec().maximum - a.tf_train_env.action_spec().minimum
  assert n + 1 == a.num_actions
  assert a.py_eval_env.timeout_factor == a.timeout_factor


if __name__ == '__main__':
  print('Testing agent ...')
  test_agent()
  print('All tests pass.')
