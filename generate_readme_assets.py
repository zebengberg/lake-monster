"""A module used to create assets for the readme."""

import os
from tf_agents.policies import random_py_policy
from environment import LakeMonsterEnvironment
from renderer import episode_as_gif


def build_random_gif():
  """Build a gif showing a random policy."""
  env = LakeMonsterEnvironment(
      monster_speed=0.7, timeout_factor=20, step_size=0.03)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_gif(py_env=env, policy=policy, filepath='assets/random.gif')


if __name__ == '__main__':
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  build_random_gif()
