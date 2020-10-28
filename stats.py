"""A class for tracking agent stats over the course of training."""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

FILE_NAME = 'stats.json'


class Stats:
  """Load and dump, add, and plot accumulated training stats."""

  def __init__(self):
    if os.path.exists(FILE_NAME):
      with open(FILE_NAME) as f:
        self.data = json.load(f)
    else:
      self.data = []

  def add(self, d):
    """Add a new dictionary to data."""
    self.data.append(d)

  def plot(self):
    """Plot all stats."""
    df = pd.DataFrame(s.data)
    df.plot(x='episode', y='reward')
    plt.show()
    # comment code below plots several metrics
    # colors = ('indianred', 'teal')
    # metrics = ('reward', 'n_env_steps')
    # fig, org_ax = plt.subplots()
    # axes = [org_ax, org_ax.twinx()]
    # for ax, color, metric in zip(axes, colors, metrics):
    #   df.plot(x='episode', y=metric, ax=ax, color=color, legend=False)
    # plt.show()

  def save(self):
    """Save data to disk."""

    # making backup copy first
    backup = 'backup' + FILE_NAME
    if os.path.exists(FILE_NAME):
      os.rename(FILE_NAME, backup)

    # writing data to disk
    with open(FILE_NAME, 'w') as f:
      json.dump(self.data, f)

    # cleanup
    if os.path.exists(backup):
      os.remove(backup)


if __name__ == '__main__':
  # plotting existing stats
  s = Stats()
  s.plot()
