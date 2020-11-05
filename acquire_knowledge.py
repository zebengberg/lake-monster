"""A script for training the RL agent."""

import os
import uuid
import webbrowser
import tensorboard
from agent import Agent
from test_agent import test_agent, log_graph
from param_search import get_random_params, log_params, log_uid, read_params


def launch_tb():
  """Launch tensorboard in a new browser tab."""
  print('Launching tensorboard. It will open shortly in a browser tab ...')
  tb = tensorboard.program.TensorBoard()
  tb.configure(logdir='logs')
  url = tb.launch()
  webbrowser.open_new_tab(url)


handmade_params = {'num_actions': 16,
                   'initial_step_size': 0.3,
                   'initial_monster_speed': 1.0,
                   'timeout_factor': 3,
                   'fc_layer_params': (20, 20),
                   'dropout_layer_params': (0.1, 0.1),
                   'learning_rate': 0.01,
                   'epsilon_greedy': 0.2,
                   'penalty_per_step': 0.0}


def build_new_agent(params=None):
  """Build new agent from scratch."""
  uid = str(uuid.uuid1().int)
  if params is None:
    params = get_random_params()

  test_agent(params)
  log_graph(params)
  log_uid(uid)
  log_params(uid, params)
  return Agent(uid, **params)


def restore_existing_agent():
  """Build new agent from scratch."""

  if not os.path.exists('agent_id.txt'):
    raise FileNotFoundError('Cannot find agent_id.txt')
  uid, params = read_params()
  test_agent(params)
  return Agent(uid, **params)


def acquire_knowledge(params=None):
  """Load an agent and train."""
  launch_tb()
  if os.path.exists('agent_id.txt'):
    a = restore_existing_agent()
  else:
    a = build_new_agent(params)

  a.train_ad_infinitum()


if __name__ == '__main__':
  acquire_knowledge(handmade_params)
