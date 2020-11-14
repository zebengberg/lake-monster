"""A module to create assets for the readme."""

import os
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy, scripted_py_policy
from environment import LakeMonsterEnvironment
from renderer import episode_as_gif


def create_random_gif():
  """Create a gif showing a random policy."""
  env = LakeMonsterEnvironment(
      monster_speed=0.7, timeout_factor=20, step_size=0.03, num_actions=8)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_gif(py_env=env, policy=policy, filepath='assets/random.gif')


def create_action_gif():
  """Create a gif showing discretization of agent actions."""
  num_actions = 8  # must be even
  env = LakeMonsterEnvironment(monster_speed=0.0, timeout_factor=8,
                               step_size=0.5, num_actions=num_actions)
  action_script = []
  for i in range(num_actions):
    action_script.append((1, i))  # step forward
    action_script.append((1, (i + num_actions // 2) % num_actions))  # back
  policy = scripted_py_policy.ScriptedPyPolicy(
      time_step_spec=None, action_spec=env.action_spec(), action_script=action_script)
  episode_as_gif(py_env=env, policy=policy,
                 filepath='assets/actions.gif', fps=1)


def create_policy_gif(policy_path=None, asset_path=None, new_params=None):
  """Create a gif showing a saved policy in action."""

  if policy_path is None:  # choosing any available policy
    policies = os.listdir('policies')
    if policies:
      policy_path = os.path.join('policies', policies[0])
    else:
      raise FileNotFoundError('No policies to show.')

  policy = tf.saved_model.load(policy_path)
  env_params = policy.get_metadata()
  for k, v in env_params.items():
    # casting from tf.Variable to python native
    env_params[k] = v.numpy().item()

  # overwriting some parameters
  if new_params:
    for k, v in new_params.items():
      env_params[k] = v

  print('Creating a gif with environment parameters:')
  print(env_params)

  py_env = LakeMonsterEnvironment(**env_params)
  tf_env = tf_py_environment.TFPyEnvironment(py_env)
  if asset_path is None:
    asset_path = 'assets/policy.gif'
  episode_as_gif(py_env, policy, asset_path, tf_env=tf_env, fps=10)


if __name__ == '__main__':
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  # create_random_gif()
  # create_action_gif()

  p_path = 'policies/87497411048514251456633633962304499656-83200'
  a_path = 'assets/capture.gif'
  params = {'step_size': 0.02, 'monster_speed': 3.8}
  create_policy_gif(p_path, a_path, params)

  p_path = 'policies/87497411048514251456633633962304499656-172100'
  a_path = 'assets/strong.gif'
  params = {'step_size': 0.01, 'monster_speed': 4.2}
  create_policy_gif(p_path, a_path, params)
