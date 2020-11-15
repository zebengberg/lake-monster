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
  webbrowser.open_new_tab(url)


def build_new_agent(use_random=False):
  """Build new agent from scratch."""
  uid = str(uuid.uuid1().int)
  if use_random:
    params = get_random_params()
  else:
    params = {}

  # TODO: fix this!!!
  # test_agent(params)
  # log_graph(params)
  log_uid(uid)
  log_params(uid, params)
  return Agent(uid, **params)


def restore_existing_agent():
  """Build new agent from scratch."""
  if not os.path.exists('agent_id.txt'):
    raise FileNotFoundError('Cannot find agent_id.txt')
  uid, params = read_params()
  # TODO: fix this
  # test_agent(params)
  return Agent(uid, **params)


if __name__ == '__main__':
  launch_tb()
  if os.path.exists('agent_id.txt'):
    a = restore_existing_agent()
  else:
    a = build_new_agent()
  a.train_ad_infinitum()
