"""Read logs created for TensorBoard by tf.summary."""

import tensorflow as tf
import glob
import pandas as pd


def build_df():
  """Use tf logs to build DataFrame containing monster speeds."""
  log_data = glob.glob('logs/events*')
  speeds = []
  episodes = []
  scores = []
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
        if v.tag == 'learning_score':
          score = tf.make_ndarray(v.tensor).item()
          scores.append(score)

  df = pd.DataFrame({'episode': episodes, 'speed': speeds, 'score': scores})
  df.set_index('episode', inplace=True)
  df.sort_index(inplace=True)
  return df


def is_progress_made():
  """Determine if any progress has been made over last 10_000 episodes."""
  num_episodes = 10_000
  df = build_df()
  current_episode = df.index[-1]
  if current_episode <= num_episodes:
    return True
  previous_episode = current_episode - num_episodes
  scores = df['score'].loc[previous_episode:current_episode]
  return scores.sum() > 0
