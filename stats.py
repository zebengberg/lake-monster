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

    self.df = None
    self.jumps = None

  def add(self, d):
    """Add a new dictionary to data."""
    self.data.append(d)

  def get_average_reward(self, monster_speed, num_evals=20):
    """Get average reward over all recent evals at specified monster speed."""
    rewards = [d['reward'] for d in self.data
               if d['monster_speed'] == monster_speed]
    rewards = rewards[-num_evals:]
    if len(rewards) == num_evals:
      return round(sum(rewards) / num_evals, 3)
    return 0.0

  def get_number_episodes_on_lt(self, monster_speed):
    """Return the number of episodes spent within this learning target."""
    for d in self.data:
      if d['monster_speed'] == monster_speed:
        return self.data[-1]['episode'] - d['episode'] + 10
    return 0

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

  def build_df(self):
    """Build common dataframe used to plot statistics."""
    df = pd.DataFrame(self.data)
    df['log_loss'] = np.log(df['loss'])
    grouped = df.groupby(by='monster_speed', as_index=False)
    for _, g in grouped:
      df.loc[g.index, 'expanding_reward'] = g['reward'].expanding().mean()
      df.loc[g.index, 'expanding_steps'] = g['n_env_steps'].expanding().mean()
      df.loc[g.index, 'expanding_log_loss'] = df['log_loss'].expanding().mean()
    self.df = df

    # jumps holds the episode in which the monster's speed has increased
    jumps = df[df['monster_speed'].diff() > 0.01]['episode']
    self.jumps = jumps

  def plot_recent_stats(self, num_recent=100000):
    """Plot expanding reward, steps, and log loss over recent episodes."""
    _, ax = plt.subplots()
    last_episode = self.df['episode'].iloc[-1]
    sliced = self.df[self.df['episode'] >= last_episode - num_recent]
    sliced.plot(x='episode', y='expanding_reward', ax=ax, linewidth='2')
    sliced.plot(x='episode', y='expanding_steps', ax=ax, linewidth='2')
    sliced.plot(x='episode', y='expanding_log_loss', ax=ax, linewidth='2')

    sliced_jumps = [l for l in self.jumps if l > sliced['episode'].iloc[0]]
    for l in sliced_jumps:
      ax.axvline(l, color='gray', alpha=0.5)
    plt.show()

  def plot_learning(self):
    """Plot monster speed and reward over time."""
    _, ax = plt.subplots()
    self.df.plot(x='episode', y='monster_speed', ax=ax)
    plt.show()

  def plot_weights(self):
    """Plot weights over time."""
    weights = [item['weights'] for item in self.data]
    weights = pd.DataFrame(weights)

    env_state_index = [t[0] for t in weights.columns]
    # pandas doesn't like tuples as column names
    weights.columns = range(len(weights.columns))

    keys = weights.columns
    weights['episode'] = [item['episode'] for item in self.data]

    _, ax = plt.subplots()
    for k in keys:
      weights.plot(x='episode', y=k, ax=ax, linewidth=1)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, env_state_index, title='env state index')
    for l in self.jumps:
      ax.axvline(l, color='gray', alpha=0.5)
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
  s.build_df()
  s.plot_recent_stats()
  s.plot_learning()
  s.plot_weights()
