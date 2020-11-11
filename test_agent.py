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
  assert len(a.q_net.layers) == 2

  assert a.agent._optimizer.get_config()['learning_rate'] == a.learning_rate

  # layers is list of all layers except for final output layer
  layers = a.q_net.layers[0].get_weights()
  for i, l in enumerate(layers):
    if i % 2 == 0:
      assert l.shape[1] == a.fc_layer_params[i // 2]
  final_layer = a.q_net.layers[1].get_weights()
  assert final_layer[0].shape[1] == a.num_actions

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
