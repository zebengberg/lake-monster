"""A script for training the agent."""

import sys
import os
import shutil
import uuid
import webbrowser
import datetime
import tensorboard
from agent import Agent
from test_agent import test_agent, log_graph
from utils import get_random_params, log_params, log_uid, read_params


def launch_tb(uid):
  """Launch tensorboard in a new browser tab."""
  print('Launching tensorboard. It will open shortly in a browser tab ...')
  tb = tensorboard.program.TensorBoard()
  tb.configure(logdir='logs/' + uid)
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
    print('Initializing new agent with parameters:')
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
  if os.path.exists('agent_id.txt'):
    os.remove('agent_id.txt')


def clear_all_knowledge():
  """Call clear_knowledge then remove policies and results."""
  if input('Do you want to clear all knowledge and statistics? (y/n) ') == 'y':
    clear_knowledge()
    if os.path.exists('policies/'):
      shutil.rmtree('policies/')
    if os.path.exists('results.json'):
      os.remove('results.json')
    if os.path.exists('logs/'):
      shutil.rmtree('logs/')


def confirm_new():
  """Ask the user to confirm initialization of new agent."""
  if os.path.exists('agent_id.txt'):
    if input('Clear partially trained agent? (y/n) ') == 'y':
      clear_knowledge()
      return True
    return False
  return True


def generate_random():
  """Train new random agent."""
  if confirm_new():
    a = build_new_agent(True)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def generate_default():
  """Train new default agent."""
  if confirm_new():
    a = build_new_agent(False)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def run_many_trainings():
  """Run a new training with random parameters every 24 hours."""
  while True:
    print('#' * 65)
    print('Training new agent!')
    print('#' * 65)
    now = datetime.datetime.now()
    duration = datetime.timedelta(hours=24)
    end_time = now + duration

    def callback():
      now = datetime.datetime.now()
      return now > end_time

    a = build_new_agent(True)
    launch_tb(a.get_uid())
    a.summative_callback = callback
    a.train_ad_infinitum()
    clear_knowledge()


def parse_args():
  """Parse command line arguments."""

  args = sys.argv
  if len(args) > 2:
    raise ValueError('Only expecting a single command line argument.')

  if len(args) == 2:
    arg = args[1]
    arg_dict = {'random': generate_random,
                'default': generate_default,
                'clear': clear_knowledge,
                'clearall': clear_all_knowledge,
                'many': run_many_trainings}

    if arg not in arg_dict:
      raise ValueError(
          f"Expecting an argument from: {', '.join(arg_dict.keys())}")
    arg_dict[arg]()

  else:

    if os.path.exists('agent_id.txt'):
      a = restore_existing_agent()
      launch_tb(a.get_uid())
    else:
      a = build_new_agent()
    a.train_ad_infinitum()


if __name__ == '__main__':
  parse_args()
