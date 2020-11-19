"""A script for training the agent."""

import sys
import os
import shutil
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
  if os.path.exists('agent_id.txt'):
    raise ValueError('Found partially trained agent already!')
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


def clear_knowledge():
  """Remove knowledge saved during last agent train."""
  if os.path.exists('videos/'):
    shutil.rmtree('videos/')
  if os.path.exists('checkpoints/'):
    shutil.rmtree('checkpoints/')
  if os.path.exists('logs/'):
    shutil.rmtree('logs/')
  if os.path.exists('agent_id.txt'):
    os.remove('agent_id.txt')


def clear_all_knowledge():
  """Call clear_knowledge then remove policies and results."""
  if input('You sure? Press `y` or `n` then hit enter. ') == 'y':
    clear_knowledge()
    if os.path.exists('policies/'):
      shutil.rmtree('policies/')
    if os.path.exists('results.json'):
      os.remove('results.json')


def confirm_new():
  """Ask the user to confirm initialization of new agent."""
  if os.path.exists('agent_id.txt'):
    if input('Clear partially trained agent? Press `y` or `n` then hit enter. ') == 'y':
      return True
    return False
  return True


def generate_random():
  """Train new random agent."""
  if confirm_new():
    launch_tb()
    a = build_new_agent(True)
    a.train_ad_infinitum()


def generate_default():
  """Train new default agent."""
  if confirm_new():
    launch_tb()
    a = build_new_agent(False)
    a.train_ad_infinitum()


def parse_args():
  """Parse command line arguments."""

  args = sys.argv
  if len(args) > 2:
    raise ValueError('Only expecting a single command line argument.')

  elif len(args) == 2:
    arg = args[1]
    arg_dict = {'random': generate_random,
                'default': generate_default,
                'clear': clear_knowledge,
                'clearall': clear_all_knowledge}

    if arg not in arg_dict:
      raise ValueError(
          f"Expecting an argument from: {', '.join(arg_dict.keys())}")
    arg_dict[arg]()

  else:
    launch_tb()
    if os.path.exists('agent_id.txt'):
      a = restore_existing_agent()
    else:
      a = build_new_agent()
    a.train_ad_infinitum()


if __name__ == '__main__':
  parse_args()
