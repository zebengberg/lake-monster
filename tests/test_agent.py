"""Test the Agent class and log tf graph information."""

import os
import shutil
from lake_monster.agent import verify
from lake_monster import configs, utils


def test_agent():
  """Run test_agent over several iterations."""

  # default parameters
  print('Testing an agent with default parameters')
  uid = 'test_agent'
  params = {'use_checkpointer': False}
  verify.verify_agent(uid, params)
  verify.log_graph(uid, write_logs=False)
  print('\n' + '#' * 65 + '\n')

  # random parameters
  for _ in range(9):
    rand_params = utils.get_random_params()
    rand_params['use_checkpointer'] = False
    print(f'Testing an agent with parameters: {rand_params}')
    verify.verify_agent(uid, rand_params)
    verify.log_graph(uid, rand_params, False)
    print('\n' + '#' * 65 + '\n')

  # cleaning up
  path = os.path.join(configs.LOG_DIR, 'test_agent')
  shutil.rmtree(path)
