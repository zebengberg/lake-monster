"""A module for evaluating policies."""

from tqdm import tqdm
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from environment import LakeMonsterEnvironment


def evaluate_episode(policy, env_params):
  """Use naive while loop to evaluate policy in single episode."""
  py_env = LakeMonsterEnvironment(**env_params)
  tf_env = TFPyEnvironment(py_env)
  ts = tf_env.reset()
  n_steps = 0
  while not ts.is_last():
    action = policy.action(ts)
    ts = tf_env.step(action.action)
    n_steps += 1

  reward = ts.reward.numpy().item()
  return reward, n_steps


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
