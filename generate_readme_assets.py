"""A module to create assets for the readme."""

import os
from tf_agents.policies import random_py_policy, scripted_py_policy
from environment import LakeMonsterEnvironment
from renderer import episode_as_gif


def build_random_gif():
  """Build a gif showing a random policy."""
  env = LakeMonsterEnvironment(
      monster_speed=0.7, timeout_factor=20, step_size=0.03, num_actions=8)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_gif(py_env=env, policy=policy, filepath='assets/random.gif')


def build_action_gif():
  """Build a gif showing discretization of agent actions."""
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


def show_path():
  pass


if __name__ == '__main__':
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  build_random_gif()
  build_action_gif()
