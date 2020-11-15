"""A module for testing the Agent class and logging tf graph information."""


import tensorflow as tf
from agent import Agent
from utils import get_random_params


class ModelWrapper(tf.Module):
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


def log_graph(uid, params=None, write_logs=True):
  """Call tf.summary.trace"""
  if params is None:
    params = {}
  a = Agent(uid, **params)

  model = ModelWrapper(a.q_net.copy(), a.use_categorical)

  x = a.tf_eval_env.reset().observation  # input to model.__call__
  if write_logs:
    summary_writer = tf.summary.create_file_writer('logs/')
    summary_writer.set_as_default()
    tf.summary.trace_on()
    model(x)  # ignoring output
    tf.summary.trace_export(name='q_net', step=0)
  else:
    model(x)  # ignoring output


def test_agent(uid, params=None):
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
    assert weights[-2].shape[1] == a.num_actions * a.q_net.num_atoms
  else:
    assert weights[-2].shape[1] == a.num_actions

  # agent parameters
  assert a.agent._epsilon_greedy == a.epsilon_greedy
  assert a.agent.train_sequence_length == a.n_step_update + 1

  # environment parameters
  n = a.tf_train_env.action_spec().maximum - a.tf_train_env.action_spec().minimum
  assert a.num_actions == n + 1
  assert a.monster_speed.numpy().item() == a.py_eval_env.monster_speed
  assert a.timeout_factor == a.py_eval_env.timeout_factor
  assert a.num_actions == a.py_eval_env.num_actions
  assert a.step_size.numpy().item() == a.py_eval_env.step_size
  assert a.use_mini_rewards == a.py_eval_env.use_mini_rewards
  assert a.use_cartesian == a.py_eval_env.use_cartesian

  for _ in range(10):
    a.driver.run()


if __name__ == '__main__':
  print('Testing an agent with default parameters')
  name = 'test_agent'
  test_agent(name)
  log_graph(name, write_logs=False)
  print('\n' + '#' * 65 + '\n')

  for _ in range(9):
    rand_params = get_random_params()
    print(f'Testing an agent with parameters: {rand_params}')
    test_agent(name, rand_params)
    log_graph(name, rand_params, False)
    print('\n' + '#' * 65 + '\n')

  print('All tests pass.')
