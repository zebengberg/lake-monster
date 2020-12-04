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
from variations import JumpingEnvironment, MultiMonsterEnvironment


# suppressing some annoying warnings
warnings.filterwarnings('ignore', category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def print_all_uuids():
  """Print list of all saved policy UUIDS along with last saved episode."""
  d = os.listdir('policies')
  d = [f.split('-')[0] for f in d]
  d = set(d)
  for uid in d:
    try:
      uid = str(int(uid))
      g = glob.glob('policies/' + uid + '*')
      g = [f.split('-')[1] for f in g]
      g = [int(f) for f in g]
      print(uid, max(g))
    except ValueError:
      continue


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


def episode_as_gif(py_env, policy, save_path, fps=10, show_path=True):
  """Create gif through py_environment render method."""

  tf_env = TFPyEnvironment(py_env)
  path = []
  with imageio.get_writer(save_path, mode='I', fps=fps) as gif:
    time_step = tf_env.reset()
    # using the policy_state to deal with scripted_policy possibility
    policy_state = policy.get_initial_state(batch_size=1)
    gif.append_data(py_env.render())
    while not time_step.is_last():
      action = policy.action(time_step, policy_state)
      time_step = tf_env.step(action.action)
      im, real_position = py_env.render('return_real')
      path.append(real_position)
      if show_path:
        im = render_agent_path(im, path)
      policy_state = action.state
      gif.append_data(np.array(im))
    for _ in range(fps):  # play for 1 more seconds
      gif.append_data(py_env.render())
  pygifsicle.optimize(save_path)


def create_random_gif():
  """Create a gif showing a random policy."""
  env_params = {'monster_speed': 0.7, 'timeout_factor': 20,
                'step_size': 0.05, 'n_actions': 8}
  py_env = LakeMonsterEnvironment(**env_params)
  policy = random_py_policy.RandomPyPolicy(time_step_spec=None,
                                           action_spec=py_env.action_spec())
  episode_as_gif(py_env, policy, save_path='assets/random.gif')


def create_action_gif():
  """Create a gif showing discretization of agent actions."""
  n_actions = 8  # should be even
  env_params = {'monster_speed': 0.0, 'timeout_factor': 8,
                'step_size': 0.5, 'n_actions': n_actions}
  py_env = LakeMonsterEnvironment(**env_params)
  action_script = [(1, 0), (1, n_actions // 2)]
  for _ in range(n_actions - 1):
    action_script.append((1, 1))  # step forward
    action_script.append((1, n_actions // 2))  # back
  policy = scripted_py_policy.ScriptedPyPolicy(time_step_spec=None,
                                               action_spec=py_env.action_spec(),
                                               action_script=action_script)
  episode_as_gif(py_env, policy, 'assets/actions.gif', fps=1, show_path=False)


def create_many_policy_gif(uid, file_path, monster_speed=4.0):
  """Create a gif superimposing the actions of many policies."""
  n_steps = 300  # = timeout_factor / step_size
  step_size = 0.01
  fps = 10
  p_paths = glob.glob('policies/' + uid + '*')

  all_positions = []
  colors = []
  for p_path in tqdm(p_paths):
    color = (np.random.randint(256), np.random.randint(128), 0)
    policy = tf.saved_model.load(p_path)
    env_params = policy.get_metadata()
    env_params = tf_to_py(env_params)

    # overwriting parameters
    env_params['step_size'] = step_size
    env_params['monster_speed'] = monster_speed
    py_env = LakeMonsterEnvironment(**env_params)
    tf_env = TFPyEnvironment(py_env)

    time_step = tf_env.reset()
    agent_positions = {}
    for step in range(n_steps):
      if not time_step.is_last():
        action = policy.action(time_step)
        time_step = tf_env.step(action.action)
      theta = py_env.total_monster_rotation - py_env.total_agent_rotation
      c, s = np.cos(theta), np.sin(theta)
      rot_matrix = np.array(((c, -s), (s, c)))
      agent_positions[step] = np.dot(rot_matrix, np.array((py_env.r, 0)))
    all_positions.append(agent_positions)
    colors.append(color)

  with imageio.get_writer(file_path, mode='I', fps=fps) as gif:
    for step in range(n_steps):
      positions = [item[step] for item in all_positions]
      im = render_many_agents(positions, colors, step,
                              step_size, 4, monster_speed)
      gif.append_data(np.array(im))
  pygifsicle.optimize(file_path)


def explore_policies():
  """Explore strong policies."""
  paths = ['28000982797640512868211384605769524580-510000',
           '185576402529663235867808454191497667940-256000',
           '150781952464835427521861702579725067620-1086000',
           '131627975635229415516782144963914819618-478000',
           '221793556195041833641466442935390234658-506000',
           '64555185743592193537153174932120725538-120000',
           '59662670505391344997989338143708473378-362000',
           '36716354962105796536622825194778333824-350000',
           '80732143607727547094796928126813095227-40000',
           '153764045992652324360525882648219764027-112000',
           '198118059822011761597806885923646036845-88000',
           '207187742707064940724130546035145792365-186000']

  paths = ['policies/' + p for p in paths]
  for i, p_path in enumerate(paths):
    if os.path.exists(p_path):
      policy = tf.saved_model.load(p_path)
      env_params = policy.get_metadata()
      env_params = tf_to_py(env_params)
      env_params['step_size'] = 0.01
      env_params['monster_speed'] = 4.3
      py_env = LakeMonsterEnvironment(**env_params)
      save_path = f'temp/gif_{i}.gif'
      episode_as_gif(py_env, policy, save_path)


def create_assets():
  """Create readme assets with specific saved policies."""
  # creating gifs for the readme
  if not os.path.exists('assets/'):
    os.mkdir('assets')
  if not os.path.exists('temp/'):
    os.mkdir('temp')
  create_random_gif()
  create_action_gif()
  uid = '150781952464835427521861702579725067620'
  file_path = 'assets/many1.gif'
  create_many_policy_gif(uid, file_path)
  uid = '131627975635229415516782144963914819618'
  file_path = 'assets/many2.gif'
  create_many_policy_gif(uid, file_path)
  uid = '80732143607727547094796928126813095227'
  file_path = 'assets/many3.gif'
  create_many_policy_gif(uid, file_path, 2.5)
  explore_policies()

  p_path = 'policies/65601302196810597370436998403635834824-12000'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    py_env = LakeMonsterEnvironment(step_size=0.02, monster_speed=3.7,
                                    use_mini_rewards=True)
    save_path = 'assets/reward.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)

  p_path = 'policies/87497411048514251456633633962304499656-83200'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    py_env = LakeMonsterEnvironment(step_size=0.02, monster_speed=3.75)
    save_path = 'assets/capture.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)

  p_path = 'policies/87497411048514251456633633962304499656-172100'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    py_env = LakeMonsterEnvironment(step_size=0.01, monster_speed=4.2)
    save_path = 'assets/strong.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)

  p_path = 'policies/28000982797640512868211384605769524580-510000'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    env_params = policy.get_metadata()
    env_params = tf_to_py(env_params)
    env_params['step_size'] = 0.001
    env_params['monster_speed'] = 4.52
    py_env = LakeMonsterEnvironment(**env_params)
    save_path = 'assets/best1.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path, fps=60)

    env_params['step_size'] = 0.003
    env_params['monster_speed'] = 4.4
    py_env = LakeMonsterEnvironment(**env_params)
    save_path = 'assets/best2.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path, fps=30)

    env_params['step_size'] = 0.02
    env_params['monster_speed'] = 4.3
    py_env = LakeMonsterEnvironment(**env_params)
    save_path = 'assets/best3.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)

  p_path = 'policies/87497411048514251456633633962304499656-136600'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    py_env = LakeMonsterEnvironment(step_size=0.01, monster_speed=4.1)
    save_path = 'assets/missteps.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)

  p_path = 'policies/36716354962105796536622825194778333824-350000'
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    env_params = policy.get_metadata()
    env_params = tf_to_py(env_params)
    env_params['timeout_factor'] = 5.0
    env_params['monster_speed'] = 4.1
    env_params['step_size'] = 0.02
    py_env = JumpingEnvironment(**env_params)
    for i in range(10):
      save_path = f'temp/jump_{i}.gif'
      print('Creating ' + save_path)
      episode_as_gif(py_env, policy, save_path)

  p_path = glob.glob('policies/multi_policies/*-450000')[0]
  if os.path.exists(p_path):
    policy = tf.saved_model.load(p_path)
    env_params = policy.get_metadata()
    env_params = tf_to_py(env_params)
    env_params['monster_speed'] = 4.05
    env_params['step_size'] = 0.01
    py_env = MultiMonsterEnvironment(**env_params)
    save_path = 'assets/multi.gif'
    print('Creating ' + save_path)
    episode_as_gif(py_env, policy, save_path)


if __name__ == '__main__':
  print_all_uuids()
  create_assets()
