"""A module for parameter search, recording results, and reading tf logs."""

import os
import random
import json
import glob
import tensorflow as tf
import pandas as pd

# sorted from least complexity to most complexity
param_universe = {
    # environment params
    'num_actions': [4, 8, 16, 32],
    'initial_step_size': [0.4, 0.2, 0.1, 0.05],
    'initial_monster_speed': [2.5, 3.0, 3.5, 4.0],
    'timeout_factor': [1.5, 2.0, 2.5, 3.0],
    'use_mini_rewards': [False, True],
    'use_cartesian': [False, True],
    'use_noisy_start': [False, True],

    # agent params
    'fc_layer_params': [(10, 10), (20, 20), (50, 50), (100, 100), (200, 200)],
    'dropout_layer_params': [None, (0.1, 0.1), (0.4, 0.4)],
    'learning_rate': [0.001, 0.0005, 0.0002],
    'epsilon_greedy': [0.3, 0.1, 0.03],
    'n_step_update': [1, 5, 10],
    'use_categorical': [False, True],
    'use_step_schedule': [False, True],
    'use_mastery': [False, True],
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
  data[uid] = {'params': params, 'results': []}

  with open('results.json', 'w') as f:
    json.dump(data, f, indent=2)

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
  data[uid]['results'].append(results)

  with open('results.json', 'w') as f:
    json.dump(data, f, indent=2)
  # remove backup copy
  os.remove('backup.json')


def merge_results_and_policies(new_results_path, new_policies_path):
  pass
# TODO: write this


def tf_to_py(d):
  """Convert a dict of tf.Variables to Python native types."""
  if not isinstance(d, dict):
    raise NotImplementedError('Only implemented for dictionaries.')
  for k, v in d.items():
    d[k] = v.numpy().item()
  return d


def py_to_tf(d):
  """Convert a dict of python variables to tf.Variables."""
  if not isinstance(d, dict):
    raise NotImplementedError('Only implemented for dictionaries.')
  for k, v in d.items():
    d[k] = tf.Variable(v)
  return d


def build_df_from_tf_logs():
  """Use tf.summary logs to build DataFrame containing monster speeds."""
  tf.summary.flush()  # flushing anything that hasn't been written
  log_data = glob.glob('logs/events*')
  speeds = {}
  scores = {}
  sizes = {}
  for path in log_data:
    data_from_path = tf.compat.v1.train.summary_iterator(path)
    for e in data_from_path:
      for v in e.summary.value:
        if v.tag == 'monster_speed':
          speed = tf.make_ndarray(v.tensor).item()
          step = e.step
          speeds[step] = speed
        elif v.tag == 'learning_score':
          score = tf.make_ndarray(v.tensor).item()
          step = e.step
          scores[step] = score
        elif v.tag == 'step_size':
          size = tf.make_ndarray(v.tensor).item()
          step = e.step
          sizes[step] = size

  df = pd.DataFrame({'speed': speeds, 'score': scores, 'step_size': sizes})
  df.sort_index(inplace=True)
  return df
