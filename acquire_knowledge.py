"""A script for training the RL agent."""

import os
import uuid
import webbrowser
import tensorboard
from agent import Agent
from test_agent import test_agent, log_graph
from utils import get_random_params, log_params, log_uid, read_params


def launch_tb():
  """Launch tensorboard in a new browser tab."""
  print('Launching tensorboard. It will open shortly in a browser tab ...')
  tb = tensorboard.program.TensorBoard()
  tb.configure(logdir='logs')
  url = tb.launch()
  url += '#scalars&_smoothingWeight=0.95'
  webbrowser.open_new_tab(url)


def build_new_agent(use_random=False):
  """Build new agent from scratch."""
  uid = str(uuid.uuid1().int)
  if use_random:
    params = get_random_params()
    print('Initializing new agent with parametes:')
    print(params)
    print('')
  else:  # using default agent parameters
    print('Initializing new agent with default parameters.\n')
    params = {}

  test_agent(uid, params)
  log_graph(uid, params)
  log_uid(uid)
  log_params(uid, params)
  return Agent(uid, **params)


def restore_existing_agent():
  """Build new agent from scratch."""
  print('Loading existing agent from file.\n')
  uid, params = read_params()
  print('Restored agent has parameters:')
  print(params)
  print('')
  test_agent(uid, params)
  return Agent(uid, **params)


if __name__ == '__main__':
  launch_tb()
  if os.path.exists('agent_id.txt'):
    a = restore_existing_agent()
  else:
    a = build_new_agent()
  a.train_ad_infinitum()
