"""Test the Agent class and log tf graph information."""

import os
from lake_monster.utils import get_random_params
from lake_monster.agent import verify
from lake_monster import configs


def test_agent():
  """Run test_agent over several iterations."""

  if os.path.exists(configs.AGENT_ID_PATH):
    raise NotImplementedError(
        'Can only test when no partially trained agent exists.')
  print('Testing an agent with default parameters')
  name = 'test_agent'
  verify.verify_agent(name)
  verify.log_graph(name, write_logs=False)
  print('\n' + '#' * 65 + '\n')

  for _ in range(9):
    rand_params = get_random_params()
    print(f'Testing an agent with parameters: {rand_params}')
    verify.verify_agent(name, rand_params)
    verify.log_graph(name, rand_params, False)
    print('\n' + '#' * 65 + '\n')
