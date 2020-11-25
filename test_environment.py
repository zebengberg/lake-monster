"""Test the environment module."""


from tqdm import tqdm
import numpy as np
import imageio
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from environment import LakeMonsterEnvironment
from animate import episode_as_video
from variations import JumpingEnvironment, MultiMonsterEnvironment


# nice environment parameters for testing with random policy
params = {'monster_speed': 0.7,
          'timeout_factor': 20,
          'step_size': 0.05,
          'n_actions': 20,
          'use_mini_rewards': True}


def validate_environment():
  """Test environment using built-in validate tool."""
  print('Validating environment ...')
  env = LakeMonsterEnvironment()
  validate_py_environment(env, episodes=50)
  print('Test successful.')


def test_py_environment_with_random(n_episodes=100):
  """Test py environment through random actions."""
  print(f'Testing py environment over {n_episodes} episodes.')
  env = LakeMonsterEnvironment(**params)

  ts = env.reset()
  rewards = []
  n_steps = []
  results = {'capture': 0, 'timeout': 0, 'success': 0}

  for _ in tqdm(range(n_episodes)):
    while not ts.is_last():
      action = np.random.randint(0, params['n_actions'])
      ts = env.step(action)

    reward = ts.reward
    rewards.append(reward)
    n_steps.append(env.n_steps)

    _, result = env.determine_reward()
    results[result] += 1
    ts = env.reset()

  assert sum(results.values()) == n_episodes
  print(f'average num of steps per episode: {np.mean(n_steps):.3f}')
  print(f'average reward per episode: {np.mean(rewards):.3f}')
  print(f"proportion of captures: {results['capture'] / n_episodes}")
  print(f"proportion of timeouts: {results['timeout'] / n_episodes}")
  print(f"proportion of successes: {results['success'] / n_episodes}")


def test_tf_environment_with_random(n_episodes=20):
  """Test tf environment through random actions."""
  print(f'Testing tf environment over {n_episodes} episodes.')
  print("You may see warnings related to TF's preferences for hardware.")
  env = LakeMonsterEnvironment(**params)
  env = TFPyEnvironment(env)
  policy = RandomTFPolicy(time_step_spec=env.time_step_spec(),
                          action_spec=env.action_spec())

  ts = env.reset()
  rewards = []
  n_steps = []

  for _ in tqdm(range(n_episodes)):
    n_step = 0
    while not ts.is_last():
      action = policy.action(ts).action
      ts = env.step(action)
      n_step += 1

    reward = ts.reward
    rewards.append(reward)
    n_steps.append(n_step)
    ts = env.reset()

  # print results
  print('average num of steps per episode:', np.mean(n_steps))
  print('average reward per episode', np.mean(rewards))


def test_video():
  """Run an episode and save video to file."""
  env = LakeMonsterEnvironment(**params)
  policy = RandomPyPolicy(time_step_spec=None, action_spec=env.action_spec())
  episode_as_video(env, policy, filename='test')


def test_movement():
  """Test handcrafted movements on fast monster."""
  print('Creating video of handcrafted strong monster.')

  env = LakeMonsterEnvironment(3.5, 3.0, 0.1, 6)
  fps = 10
  actions = [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 1, 3]
  path = 'videos/test-handcrafted.mp4'
  with imageio.get_writer(path, fps=fps) as video:
    time_step = env.reset()
    video.append_data(env.render())
    while not time_step.is_last():
      action = actions.pop()
      time_step = env.step(action)
      video.append_data(env.render())
    video.append_data(env.render())
  print(f'Video created and saved as {path}')


def test_jumping():
  """Test jumping environment."""
  print('Testing jumping environment.')
  env = JumpingEnvironment(**params)
  validate_py_environment(env, episodes=10)
  policy = RandomPyPolicy(time_step_spec=None, action_spec=env.action_spec())
  episode_as_video(env, policy, filename='test_jumping')


def test_multi_monster():
  """Test multi-monster environment."""
  print('Testing multi-monster environment.')
  env = MultiMonsterEnvironment(n_monsters=3, **params)
  validate_py_environment(env, episodes=10)
  policy = RandomPyPolicy(time_step_spec=None, action_spec=env.action_spec())
  episode_as_video(env, policy, filename='test_multi')


if __name__ == '__main__':
  print('\n' + '#' * 65)
  validate_environment()
  print('\n' + '#' * 65)
  test_py_environment_with_random()
  print('\n' + '#' * 65)
  test_tf_environment_with_random()
  print('\n' + '#' * 65)
  test_video()
  test_movement()
  test_jumping()
  test_multi_monster()
  print('\n' + '#' * 65)
  print('All tests pass.')
