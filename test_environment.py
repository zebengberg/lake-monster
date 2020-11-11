"""Test the environment module."""


from tqdm import tqdm
import numpy as np
from tf_agents.environments import utils, tf_py_environment
from tf_agents.policies import random_py_policy, random_tf_policy
import imageio
from environment import LakeMonsterEnvironment
from renderer import episode_as_video


TEST_VIDEO_FILENAME = 'test'

# nice environment parameters for testing
monster_speed = 0.7
timeout_factor = 20
step_size = 0.05
num_actions = 20
params = {'monster_speed': monster_speed, 'timeout_factor': timeout_factor,
          'step_size': step_size, 'num_actions': num_actions}


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

  ts = env.reset()
  rewards = []
  num_steps = []
  results = {'capture': 0, 'timeout': 0, 'success': 0}

  for _ in tqdm(range(num_episodes)):
    while not ts.is_last():
      action = np.random.randint(0, num_actions)
      ts = env.step(action)

    reward = ts.reward
    rewards.append(reward)
    num_steps.append(env.num_steps)

    _, result = env.determine_reward()
    results[result] += 1
    ts = env.reset()

  assert sum(results.values()) == num_episodes

  # print results
  print('average num of steps per episode:', np.mean(num_steps))
  print('average reward per episode', np.mean(rewards))
  print('proportion of timeouts', results['timeout'] / num_episodes)
  print('proportion of captures', results['capture'] / num_episodes)
  print('proportion of successes', results['success'] / num_episodes)


def test_tf_environment_with_random(num_episodes=100):
  """Test tf environment through random actions."""
  print(f'Testing tf environment with {num_episodes} episodes.')
  print("You may see warnings related to TF's preferences for hardware.")
  env = LakeMonsterEnvironment(**params)
  env = tf_py_environment.TFPyEnvironment(env)
  policy = random_tf_policy.RandomTFPolicy(
      time_step_spec=env.time_step_spec(), action_spec=env.action_spec())

  ts = env.reset()
  rewards = []
  num_steps = []

  for _ in tqdm(range(num_episodes)):
    num_step = 0
    while not ts.is_last():
      action = policy.action(ts).action
      ts = env.step(action)
      num_step += 1

    reward = ts.reward
    rewards.append(reward)
    num_steps.append(num_step)
    ts = env.reset()

  # print results
  print('average num of steps per episode:', np.mean(num_steps))
  print('average reward per episode', np.mean(rewards))


def test_video():
  """Run an episode and save video to file."""
  env = LakeMonsterEnvironment(**params)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_video(py_env=env, policy=policy, filename=TEST_VIDEO_FILENAME)


def test_movement():
  """Test strong movements on fast monster."""
  print('Creating vide of pre-programmed strong monster.')
  params = {'num_actions': 6,
            'step_size': 0.1,
            'monster_speed': 3.5,
            'timeout_factor': 3}
  env = LakeMonsterEnvironment(**params)
  fps = 10
  actions = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 2, 2, 4]
  path = 'videos/strong_test.mp4'
  with imageio.get_writer(path, fps=fps) as video:
    time_step = env.reset()
    video.append_data(env.render())
    while not time_step.is_last():
      action = actions.pop()
      time_step = env.step(action)
      video.append_data(env.render())
    video.append_data(env.render())
  print(f'Video created and saved as {path}')


if __name__ == '__main__':
  print('\n' + '#' * 80)
  validate_environment()
  print('\n' + '#' * 80)
  test_py_environment_with_random()
  print('\n' + '#' * 80)
  test_tf_environment_with_random()
  print('\n' + '#' * 80)
  test_video()
  test_movement()
  print('\n' + '#' * 80)
  print('All tests pass.')
