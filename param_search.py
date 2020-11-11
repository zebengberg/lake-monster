"""A module for parameter search and recording results."""

import os
import random
import json

# sorting from least complexity to most complexity
param_universe = {
    'num_actions': [4, 8, 12, 16],
    'initial_step_size': [0.4, 0.3, 0.2, 0.1],
    'initial_monster_speed': [1.0, 2.0, 3.0],
    'timeout_factor': [2.0, 3.0, 4.0, 5.0],
    'fc_layer_params': [(10, 10), (20, 20), (50, 50), (100, 100), (200, 200)],
    'learning_rate': [0.1, 0.01, 0.001],
    'epsilon_greedy': [0.3, 0.1, 0.03],
    'penalty_per_step': [0, 0.01, 0.1],
    'dropout_layer_params': [None, (0.1, 0.1), (0.4, 0.4)],
    'n_step_update': [1, 5, 10],
}


def get_random_params():
  """Get a dictionary of random parameters."""
  return {key: random.choice(param_universe[key]) for key in param_universe}


def log_uid(uid):
  """Log the uid of the newly instantiated agent."""
  with open('agent_id.txt', 'w') as f:
    f.write(uid)


def read_params():
  """Use 'agent_id.txt' file to read all agent parameters."""
  with open('agent_id.txt') as f:
    uid = f.read()
  with open('results.json') as f:
    data = json.load(f)
  return uid, data[uid]['params']


def log_params(uid, params):
  """Log parameters when first instantiating new agent."""
  if os.path.exists('results.json'):
    with open('results.json') as f:
      data = json.load(f)
    # making a backup copy first
    os.rename('results.json', 'backup.json')
  else:
    data = {}

  assert uid not in data
  data[uid] = {'params': params, 'results': {}}

  with open('results.json', 'w') as f:
    json.dump(data, f)

  # remove backup copy
  if os.path.exists('backup.json'):
    os.remove('backup.json')


def log_results(uid, results):
  """Record key evaluation results during training."""

  with open('results.json') as f:
    data = json.load(f)
    # making a backup copy first
  os.rename('results.json', 'backup.json')

  assert uid in data
  # possibly overwriting an existing entry
  data[uid]['results'] = results

  with open('results.json', 'w') as f:
    json.dump(data, f)
  # remove backup copy
  os.remove('backup.json')
