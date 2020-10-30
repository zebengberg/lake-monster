"""A class for tracking agent stats over the course of training."""

import os
import json
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FILE_NAME = 'stats.json'


def jsonify_dict(d):
  """Cast tuple keys to strings in order to save as json."""
  jsonified = {}
  for key in d:
    if type(key) == tuple:
      as_list = [str(i) for i in key]
      as_str = ' '.join(as_list)
      jsonified[as_str] = d[key]
    else:
      jsonified[key] = d[key]
  return jsonified


def dejsonify_dict(d):
  """Cast string keys to tuples in order to load from json."""
  dejsonified = {}
  for key in d:
    if ' ' in key:  # uses spaces as indicator of python tuple
      as_list = key.split(' ')
      as_list = [int(i) for i in as_list]
      tuple_key = tuple(as_list)
      dejsonified[tuple_key] = d[key]
    else:
      dejsonified[key] = d[key]
  return dejsonified


class Stats:
  """Load and dump, add, and plot accumulated training stats."""

  def __init__(self):
    if os.path.exists(FILE_NAME):
      print('Reading agent progress from disk ...')
      with open(FILE_NAME) as f:
        data_list = json.load(f)
        for j in data_list:
          j['weights'] = dejsonify_dict(j['weights'])
        self.data = data_list
    else:
      print('Initializing agent progress for the first time!')
      self.data = []

  def add(self, d):
    """Add a new dictionary to data."""
    self.data.append(d)

  def get_recent_average_reward(self, monster_speed, sample_size=100):
    """Return average reward over recent episodes at current monster speed."""
    if len(self.data) < sample_size:
      return 0.0
    rewards = [item['reward'] for item in self.data[-sample_size:]
               if round(item['monster_speed'], 2) == round(monster_speed, 2)]
    if len(rewards):
      return 0.0
    return sum(rewards) / sample_size

  def get_last_monster_speed(self):
    """Return last known monster speed."""
    if self.data:
      return self.data[-1]['monster_speed']
    return None

  def get_weight_indices(self):
    """Return the indices of the weights tracked."""
    if self.data:
      return list(self.data[-1]['weights'].keys())
    return None

  def plot_rewards(self):
    """Plot reward over time."""
    df = pd.DataFrame(self.data)
    _, ax = plt.subplots()
    df.plot(x='episode', y='reward', ax=ax, marker='o', linestyle='')
    df['rolling_reward'] = df['reward'].rolling(100).mean()
    df.plot(x='episode', y='rolling_reward', ax=ax, linewidth='4')
    plt.show()

  def plot_weights(self):
    """Plot weights over time."""
    weights = [item['weights'] for item in self.data]
    df = pd.DataFrame(weights)

    # pandas doesn't like tuples as column names
    env_state_index = [t[0] for t in df.columns]

    df.columns = range(len(df.columns))
    keys = df.columns
    df['episode'] = [item['episode'] for item in self.data]

    _, ax = plt.subplots()
    for k in keys:
      df.plot(x='episode', y=k, ax=ax)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, env_state_index, title='env state index')
    plt.show()

  def plot_stats(self):
    df = pd.DataFrame(self.data)
    df['rolling_steps'] = df['n_env_steps'].rolling(1000).mean() / 50
    df['rolling_reward'] = df['reward'].rolling(1000).mean()
    df['rolling_loss'] = np.log(df['loss'].rolling(1000).mean())
    _, ax = plt.subplots()
    df.plot(x='episode', y='rolling_steps', ax=ax)
    df.plot(x='episode', y='rolling_reward', ax=ax)
    df.plot(x='episode', y='rolling_loss', ax=ax)
    plt.show()

  def save(self):
    """Save data to disk."""

    # making backup copy first
    backup = 'backup' + FILE_NAME
    if os.path.exists(FILE_NAME):
      os.rename(FILE_NAME, backup)

    # writing data to disk
    with open(FILE_NAME, 'w') as f:
      jsonified = copy.deepcopy(self.data)
      for j in jsonified:
        j['weights'] = jsonify_dict(j['weights'])
      json.dump(jsonified, f)

    # cleanup
    if os.path.exists(backup):
      os.remove(backup)


if __name__ == '__main__':
  # plotting existing stats
  s = Stats()
  s.plot_stats()
  # s.plot_weights()
  # s.plot_rewards()
