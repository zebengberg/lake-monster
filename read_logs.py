"""Read logs created for TensorBoard by tf.summary."""

import tensorflow as tf
import glob
import pandas as pd
import numpy as np


def build_df():
  """Use tf logs to build DataFrame containing monster speeds."""
  log_data = glob.glob('logs/events*')
  speeds = []
  episodes = []
  scores = []
  step_sizes = []
  for path in log_data:
    # not yet implemented in v2?
    data_from_path = tf.compat.v1.train.summary_iterator(path)
    for e in data_from_path:
      for v in e.summary.value:
        if v.tag == 'monster_speed':
          speed = tf.make_ndarray(v.tensor).item()
          step = e.step
          speeds.append(speed)
          episodes.append(step)
        elif v.tag == 'learning_score':
          score = tf.make_ndarray(v.tensor).item()
          scores.append(score)
        elif v.tag == 'step_size':
          size = tf.make_ndarray(v.tensor).item()
          step_sizes.append(size)

  df = pd.DataFrame({'episode': episodes, 'speed': speeds,
                     'score': scores, 'step_size': step_sizes})
  df.set_index('episode', inplace=True)
  df.sort_index(inplace=True)
  return df


def is_progress_made():
  """Determine if any progress has been made over recent episodes."""
  df = build_df()

  initial_step_size = df['step_size'].loc[0]
  current_episode = df.index[-1]
  current_step_size = df['step_size'].loc[current_episode]

  # the log approximates how many step_size reductions have been made
  num_episodes = int(1e4 * np.log(1 + initial_step_size / current_step_size))
  print(num_episodes)

  # only looking at entries with step_size equal to current value
  df = df[df['step_size'] == current_step_size]
  initial_episode = df.index[0]
  if current_episode - initial_episode < num_episodes:
    return True
  df = df.loc[current_episode - num_episodes: current_episode]
  return df['score'].sum() > 0
