"""A module for evaluating policies."""

import json
import pandas as pd
import matplotlib.pyplot as plt
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from environment import LakeMonsterEnvironment
from variations import MultiMonsterEnvironment, JumpingEnvironment


def evaluate_episode(policy, env_params):
  """Use naive while loop to evaluate policy in single episode."""
  if 'n_monsters' in env_params:
    env = MultiMonsterEnvironment
  elif 'is_jumping' in env_params:
    env = JumpingEnvironment
  else:
    env = LakeMonsterEnvironment
  py_env = env(**env_params)
  tf_env = TFPyEnvironment(py_env)
  ts = tf_env.reset()
  n_steps = 0
  while not ts.is_last():
    action = policy.action(ts)
    ts = tf_env.step(action.action)
    n_steps += 1

  reward = ts.reward.numpy().item()
  return reward, n_steps * py_env.step_size


def probe_policy_const_steps(policy, env_params):
  """Determine the maximum monster speed at which policy can succeed."""
  highest_speed_with_success = 0.0
  n_steps_at_success = 0
  n_consecutive_fails = 0
  current_monster_speed = 0.0
  delta = 1.0

  while True:
    print('.', end='', flush=True)
    current_monster_speed += delta
    env_params['monster_speed'] = current_monster_speed
    env_params['use_mini_rewards'] = False

    reward, n_steps = evaluate_episode(policy, env_params)
    reward = round(reward)
    if reward not in [0, 1]:
      raise ValueError(f'Strange reward. Reward encountered: {reward}')

    if reward == 0:
      n_consecutive_fails += 1
    else:
      highest_speed_with_success = current_monster_speed
      n_steps_at_success = n_steps
      n_consecutive_fails = 0

    if n_consecutive_fails == 3:
      if delta < 0.001 - 1e-6:  # tolerance
        print('')
        return highest_speed_with_success, n_steps_at_success
      delta *= 0.5
      current_monster_speed = highest_speed_with_success
      n_consecutive_fails = 0


def probe_policy(policy, env_params):
  """Call probe_policy at different step_sizes."""
  current_step_size = env_params['step_size']
  result = {'monster_speed': 0.0, 'step_size': 0.0, 'n_env_steps': 0}

  for multiplier in [1/16, 1/8, 1/4, 1/2, 1]:
    step_size = multiplier * current_step_size
    env_params['step_size'] = step_size
    monster_speed, n_env_steps = probe_policy_const_steps(policy, env_params)
    if monster_speed > result['monster_speed']:
      result['monster_speed'] = monster_speed
      result['step_size'] = step_size
      result['n_env_steps'] = n_env_steps

  return result


def result_df():
  """Return DataFrame of monster speed data in results.json."""
  with open('results.json') as f:
    data = json.load(f)
  dfs = []
  params = {}
  for uid in data:
    params[uid] = data[uid]['params']
    results = data[uid]['results']
    if results:
      df = pd.DataFrame(results)
      df = df.set_index('n_episode', drop=True)
      df = df.drop(['step_size', 'n_env_steps'], axis=1)
      df = df.rename(columns={'monster_speed': uid})
      dfs.append(df)
  return pd.concat(dfs, axis=1), params


def plot_results(policies=None):
  """Plot evaluation monter speeds over training."""
  df, _ = result_df()
  if policies:
    df = df[policies]
  df = df[df.index <= 600_000]
  df = df.rolling(25).mean()
  plt.figure(figsize=(12, 8))
  df.plot(legend=False, ax=plt.gca())
  plt.xlabel('episode number')
  plt.ylabel('monster speed')
  plt.title('Smoothed evaluation scores over training')
  # plt.legend(loc='lower right', fontsize='xx-small')
  plt.grid()
  plt.savefig('assets/results.png', dpi=300)
  plt.show()


def print_strongest_policies():
  """Print a markdown table showing agent parameters and evaluation results."""

  df, params = result_df()

  shortened_names = {
      'n_actions': 'n act',
      'initial_step_size': 'init step',
      'initial_monster_speed': 'init speed',
      'timeout_factor': 'timeout',
      'fc_layer_params': 'layers',
      'dropout_layer_params': 'dropout',
      'learning_rate': 'learn rate',
      'epsilon_greedy': 'epsilon',
      'n_step_update': 'update',
      'use_categorical': 'categorical',
      'use_step_schedule': 'schedule'
  }

  params_df = []
  for p in df.columns:
    # printing out high speed policies separately from markdown
    episode = df[p].idxmax()
    speed = df[p][episode]
    if speed > 4.2:
      print(p + '-' + str(episode))

    results = {}
    results['max speed'] = round(speed, 3)
    results['avg speed'] = round(df[p].mean(), 3)
    for k, v in shortened_names.items():
      results[v] = params[p][k]

    if results['categorical']:
      results['dropout'] = 'None'
    if results['dropout'] is None:  # getting None to appear in markdown
      results['dropout'] = 'None'

    params_df.append(results)

  params_df = pd.DataFrame(params_df)
  params_df = params_df.sort_values(by='max speed', axis=0, ascending=False)

  print(params_df.to_markdown(index=False))


if __name__ == '__main__':
  print_strongest_policies()
  plot_results()
