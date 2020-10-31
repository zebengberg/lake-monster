"""Test the environment module."""


from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tf_agents.environments import utils, tf_py_environment
from tf_agents.policies import random_py_policy
from environment import LakeMonsterEnvironment
from renderer import episode_as_video

TEST_VIDEO_FILENAME = 'test.mp4'

# nice environment parameters for testing
test_monster_speed = 0.7
test_timeout_factor = 20
test_step_size = 0.05
params = {'monster_speed': test_monster_speed,
          'timeout_factor': test_timeout_factor, 'step_size': test_step_size}


def validate_environment():
  """Test environment using built-in validate tool."""
  print('Validating environment ...')
  env = LakeMonsterEnvironment()
  utils.validate_py_environment(env, episodes=10)
  print('Test successful.')


def test_py_environment_with_random(num_episodes=1000):
  """Test py environment through random actions."""
  print(f'Testing py environment with {num_episodes} episodes.')
  env = LakeMonsterEnvironment(**params)
  time_step = env.reset()
  rewards = []
  num_steps = []
  captures = []
  timeouts = []
  successes = []

  for _ in tqdm(range(num_episodes)):
    while not time_step.is_last():
      action = np.random.uniform(low=0, high=2*np.pi)
      time_step = env.step(action)

    reward = time_step.reward
    rewards.append(reward)
    num_steps.append(env.num_steps)
    captures.append(env.r >= 1.0 and reward == -1.0)
    timeouts.append(env.r < 1.0 and reward == -1.0)
    successes.append(reward == 1.0)
    time_step = env.reset()

  assert sum(captures) + sum(timeouts) + sum(successes) == num_episodes

  # print results
  print('average num of steps per episode:', np.mean(num_steps))
  print('average reward per episode', np.mean(rewards))
  print('proportion of timeouts', sum(timeouts) / num_episodes)
  print('proportion of captures', sum(captures) / num_episodes)
  print('proportion of successes', sum(successes) / num_episodes)


def test_tf_environment_with_random(num_episodes=100):
  """Test tf environment through random actions."""
  print(f'Testing tf environment with {num_episodes} episodes.')
  env = LakeMonsterEnvironment(**params)
  env = tf_py_environment.TFPyEnvironment(env)
  time_step = env.reset()

  assert env.batch_size == 1
  action = tf.random.uniform(minval=0, maxval=2*np.math.pi, shape=())
  assert env.action_spec().is_compatible_with(action)

  env.reset()
  rewards = []
  num_steps = []

  for _ in tqdm(range(num_episodes)):
    num_step = 0
    while not time_step.is_last():
      action = tf.random.uniform(minval=0, maxval=2*np.math.pi, shape=())
      time_step = env.step(action)
      num_step += 1

    reward = time_step.reward
    rewards.append(reward)
    num_steps.append(num_step)
    time_step = env.reset()

  # print results
  print('average num of steps per episode:', np.mean(num_steps))
  print('average reward per episode', np.mean(rewards))


def test_video():
  """Run an episode and save video to file."""
  env = LakeMonsterEnvironment(**params)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_video(py_env=env, policy=policy, filename=TEST_VIDEO_FILENAME)


if __name__ == '__main__':
  print('\n' + '#' * 80)
  validate_environment()
  print('\n' + '#' * 80)
  test_py_environment_with_random()
  print('\n' + '#' * 80)
  test_tf_environment_with_random()
  print('\n' + '#' * 80)
  test_video()
  print('\n' + '#' * 80)
  print('All tests pass.')
