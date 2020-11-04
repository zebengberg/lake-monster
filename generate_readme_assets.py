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


def create_policy_gif(dirpath=None):
  """Create a gif showing a saved policy in action."""

  if dirpath is None:  # choosing the first one available
    policies = os.listdir('policies')
    if policies:
      dirpath = os.path.join('policies', policies[0])
      print(dirpath)
    else:
      print('No policies to show.')

  if dirpath is not None:
    policy = tf.saved_model.load(dirpath)
    env_params = policy.get_metadata()
    for k, v in env_params.items():
      # casting from tf.Variable to python native
      env_params[k] = v.numpy().item()
    print('Creating a gif with environment parameters:')
    print(env_params)

    py_env = LakeMonsterEnvironment(**env_params)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    episode_as_gif(py_env, policy, 'assets/policy1.gif', tf_env=tf_env, fps=5)


if __name__ == '__main__':
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  create_random_gif()
  create_action_gif()
  create_policy_gif()
