"""A module for training the agent."""

import sys
import os
import glob
import uuid
import webbrowser
import datetime
import tensorboard
from lake_monster.agent.agent import Agent, MultiMonsterAgent, JumpingAgent
from lake_monster.agent.verify import verify_agent, log_graph
from lake_monster.utils import get_random_params, log_params, log_uid, read_params
from lake_monster import configs


def launch_tb(uid):
  """Launch tensorboard in a new browser tab."""
  print('Launching tensorboard. It will open shortly in a browser tab ...')
  tb = tensorboard.program.TensorBoard()
  logdir = os.path.join(configs.LOG_DIR, uid)
  tb.configure(logdir=logdir)
  url = tb.launch()
  url += '#scalars&_smoothingWeight=0.95'
  webbrowser.open_new_tab(url)


def build_new_agent(use_random=False):
  """Build new agent from scratch."""
  if os.path.exists(configs.AGENT_ID_PATH):
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

  verify_agent(uid, params)
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
  verify_agent(uid, params)
  return Agent(uid, **params)


def clear_knowledge():
  """Remove videos and checkpoints saved during last trained agent."""
  for f in glob.glob(configs.VIDEO_DIR + '/*'):
    os.remove(f)
  for f in glob.glob(configs.CHECKPOINT_DIR + '/*'):
    os.remove(f)
  if os.path.exists(p := configs.AGENT_ID_PATH):
    os.remove(p)


def clear_all_knowledge():
  """Clear all saved policies, results, logs, videos, and checkpoints."""
  if input('Do you want to clear all knowledge and statistics? (y/n) ') == 'y':
    clear_knowledge()
    if os.path.exists(p := configs.RESULTS_PATH):
      os.remove(p)
    for f in glob.glob(configs.LOG_DIR + '/*'):
      os.remove(f)
    for f in glob.glob(configs.POLICY_DIR + '/*'):
      os.remove(f)


def confirm_new():
  """Ask the user to confirm initialization of new agent."""
  if os.path.exists(configs.AGENT_ID_PATH):
    if input('Clear partially trained agent? (y/n) ') == 'y':
      clear_knowledge()
      return True
    return False
  return True


def generate_random():
  """Train a new agent with random parameters."""
  if confirm_new():
    a = build_new_agent(True)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def generate_default():
  """Train a new agent with default parameters."""
  if confirm_new():
    a = build_new_agent(False)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def generate_multi():
  """Train a MultiMonsterAgent with preset parameters."""
  if confirm_new():
    uid = str(uuid.uuid1().int)
    params = {
        'n_monsters': 6,
        'n_actions': 8,
        'initial_monster_speed': 2.0,
        'timeout_factor': 2.0}

    log_uid(uid)
    log_params(uid, params)
    a = MultiMonsterAgent(uid=uid, **params)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def generate_jump():
  """Train a JumpingAgent with default LakeMonster parameters."""
  if confirm_new():
    uid = str(uuid.uuid1().int)
    print('Initializing new JumpingAgent with default parameters.\n')
    params = {}
    verify_agent(uid, params)
    log_graph(uid, params)
    log_uid(uid)
    log_params(uid, params)
    a = JumpingAgent(uid, **params)
    launch_tb(a.get_uid())
    a.train_ad_infinitum()


def run_many_trainings():
  """Train a new agent with random parameters every 24 hours."""
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


def print_help():
  """Print this help message."""
  t = ' ' * 4
  print('Usage: python train.py [arg]')
  print('Train an RL agent to solve the lake monster problem.')
  print(t + 'The optional argument [arg] could be one of:')

  for arg, f in ARG_DICT.items():
    l = t + '--' + f'{arg: <10}' + t + f.__doc__
    print(l)


def parse_args():
  """Parse command line arguments."""

  args = sys.argv
  if len(args) > 2:
    raise ValueError('Only expecting a single command line argument.')

  if len(args) == 2:
    arg = args[1]
    args = arg.lstrip('-')

    if arg not in ARG_DICT:
      raise ValueError(f"Expecting arg from: {', '.join(ARG_DICT.keys())}")
    ARG_DICT[arg]()

  elif len(args) == 1:
    if os.path.exists(configs.AGENT_ID_PATH):
      a = restore_existing_agent()
    else:
      a = build_new_agent()
    launch_tb(a.get_uid())
    a.train_ad_infinitum()

  else:
    raise ValueError('Expecting at most one argument. Try passing --help.')


ARG_DICT = {'default': generate_default,
            'random': generate_random,
            'many': run_many_trainings,
            'multi': generate_multi,
            'jump': generate_jump,
            'clear': clear_knowledge,
            'clearall': clear_all_knowledge,
            'help': print_help}

if __name__ == '__main__':
  parse_args()
