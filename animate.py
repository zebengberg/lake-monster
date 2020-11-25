"""A module to create videos, gifs, and assets for the readme."""

import os
import glob
import warnings
import absl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import imageio
# including this import for pipreqs
import imageio_ffmpeg  # pylint: disable=unused-import
import pygifsicle
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies import random_py_policy, scripted_py_policy
from environment import LakeMonsterEnvironment
from utils import tf_to_py
from render import render_many_agents, render_agent_path


# suppressing some annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_random_gif():
  """Create a gif showing a random policy."""
  env_params = {'monster_speed': 0.7, 'timeout_factor': 20,
                'step_size': 0.03, 'n_actions': 8}
  env = LakeMonsterEnvironment(**env_params)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=env.action_spec())
  episode_as_gif(policy, env_params, filepath='assets/random.gif')


def create_action_gif():
  """Create a gif showing discretization of agent actions."""
  n_actions = 8  # should be even
  env_params = {'monster_speed': 0.0, 'timeout_factor': 8,
                'step_size': 0.5, 'n_actions': n_actions}
  env = LakeMonsterEnvironment(**env_params)
  action_script = []
  for i in range(n_actions):
    action_script.append((1, i))  # step forward
    action_script.append((1, (i + n_actions // 2) % n_actions))  # back
  policy = scripted_py_policy.ScriptedPyPolicy(
      time_step_spec=None, action_spec=env.action_spec(), action_script=action_script)
  episode_as_gif(policy, env_params, filepath='assets/actions.gif', fps=1)


def create_policy_gif(policy_path, asset_path, new_params=None):
  """Create a gif showing a saved policy in action."""

  policy = tf.saved_model.load(policy_path)
  env_params = policy.get_metadata()
  env_params = tf_to_py(env_params)

  # overwriting some parameters
  if new_params:
    for k, v in new_params.items():
      env_params[k] = v

  episode_as_gif(policy, env_params, asset_path, fps=10)


def episode_as_video(py_env, policy, filename, fps=10):
  """Create mp4 video through py_environment render method."""
  tf_env = TFPyEnvironment(py_env)

  with imageio.get_writer('tmp.mp4', fps=fps) as video:
    time_step = tf_env.reset()
    video.append_data(py_env.render())
    while not time_step.is_last():
      action = policy.action(time_step).action
      time_step = tf_env.step(action)
      video.append_data(py_env.render())
    for _ in range(3 * fps):  # play for 3 more seconds
      video.append_data(py_env.render())

  # giving video file a more descriptive name
  _, result = py_env.determine_reward()
  if not os.path.exists('videos/'):
    os.mkdir('videos/')
  filename = os.path.join('videos/', filename + '-' + result + '.mp4')
  os.rename('tmp.mp4', filename)


def episode_as_gif(policy, env_params, filepath, fps=30):
  """Create gif through py_environment render method."""

  py_env = LakeMonsterEnvironment(**env_params)
  tf_env = TFPyEnvironment(py_env)

  with imageio.get_writer(filepath, mode='I', fps=fps) as gif:
    time_step = tf_env.reset()
    # using the policy_state to deal with scripted_policy possibility
    policy_state = policy.get_initial_state(batch_size=1)
    gif.append_data(py_env.render())
    while not time_step.is_last():
      action = policy.action(time_step, policy_state)
      time_step = tf_env.step(action.action)
      policy_state = action.state
      gif.append_data(py_env.render())
    for _ in range(fps):  # play for 1 more seconds
      gif.append_data(py_env.render())
  pygifsicle.optimize(filepath)


def create_policy_gif_with_path(policy_path, asset_path, new_params=None):
  """Create a gif showing the agent's path over an episode."""

  policy = tf.saved_model.load(policy_path)
  env_params = policy.get_metadata()
  env_params = tf_to_py(env_params)

  # overwriting parameters
  if new_params:
    for k, v in new_params.items():
      env_params[k] = v

  py_env = LakeMonsterEnvironment(**env_params)
  tf_env = TFPyEnvironment(py_env)

  time_step = tf_env.reset()
  path = []
  with imageio.get_writer(asset_path, mode='I', fps=10) as gif:
    while not time_step.is_last():
      action = policy.action(time_step)
      time_step = tf_env.step(action.action)
      im, real_position = py_env.render('return_real')
      path.append(real_position)
      im = render_agent_path(im, path)
      gif.append_data(np.array(im))
    for _ in range(30):
      gif.append_data(np.array(im))
  pygifsicle.optimize(asset_path)


def create_many_policy_gif():
  """Create a gif superimposing the actions of many policies."""
  num_steps = 300  # == timeout_factor / step_size
  step_size = 0.01
  monster_speed = 4.0
  fps = 10

  uid = '65601302196810597370436998403635834824'
  policy_paths = glob.glob('policies/' + uid + '*')

  all_positions = []
  colors = []
  for policy_path in tqdm(policy_paths):
    color = (np.random.randint(256), np.random.randint(128), 0)
    policy = tf.saved_model.load(policy_path)
    env_params = policy.get_metadata()
    env_params = tf_to_py(env_params)

    # overwriting parameters
    env_params['step_size'] = step_size
    env_params['monster_speed'] = monster_speed
    py_env = LakeMonsterEnvironment(**env_params)
    tf_env = TFPyEnvironment(py_env)

    time_step = tf_env.reset()
    agent_positions = {}
    for step in range(num_steps):
      if not time_step.is_last():
        action = policy.action(time_step)
        time_step = tf_env.step(action.action)
      agent_positions[step] = py_env.position
    all_positions.append(agent_positions)
    colors.append(color)

  filepath = 'assets/many.gif'
  with imageio.get_writer(filepath, mode='I', fps=fps) as gif:
    for step in range(num_steps):
      positions = [item[step] for item in all_positions]
      im = render_many_agents(positions, colors, step,
                              step_size, 4, monster_speed)
      gif.append_data(np.array(im))
  pygifsicle.optimize(filepath)


if __name__ == '__main__':
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  create_random_gif()
  create_action_gif()
  create_many_policy_gif()

  p_path = 'policies/65601302196810597370436998403635834824-12000'
  a_path = 'assets/reward.gif'
  params = {'step_size': 0.02, 'monster_speed': 3.7, 'use_mini_rewards': True}
  create_policy_gif_with_path(p_path, a_path, params)

  p_path = 'policies/87497411048514251456633633962304499656-83200'
  a_path = 'assets/capture.gif'
  params = {'step_size': 0.02, 'monster_speed': 3.75}
  create_policy_gif_with_path(p_path, a_path, params)

  p_path = 'policies/87497411048514251456633633962304499656-172100'
  a_path = 'assets/strong.gif'
  params = {'step_size': 0.01, 'monster_speed': 4.2}
  create_policy_gif_with_path(p_path, a_path, params)

  p_path = 'policies/87497411048514251456633633962304499656-136600'
  a_path = 'assets/path.gif'
  params = {'step_size': 0.01, 'monster_speed': 4.1}
  create_policy_gif_with_path(p_path, a_path, params)
