"""Utility functions for rendering an environment and displaying an episode in video."""

import os
import numpy as np
from PIL import Image, ImageDraw
import imageio
# including this import for pipreqs
import imageio_ffmpeg  # pylint: disable=unused-import
import pygifsicle


SIZE = 480
CENTER = SIZE // 2
RADIUS = 200
RED = (250, 50, 0)


def coords_to_rect(coords):
  """Convert environment coordinates to PIL rectangle coordinates."""
  x, y = coords
  y *= -1
  x, y = CENTER + RADIUS * x, CENTER + RADIUS * y
  return x - 8, y - 8, x + 8, y + 8


def angle_to_rect(angle):
  """Convert environment angle to PIL rectangle coordinates."""
  x, y = np.cos(angle), np.sin(angle)
  return coords_to_rect((x, y))


def vector_to_rect(vector):
  """Convert action vector to PIL rectangle coordinates."""
  x, y = 50 * vector
  u, v = CENTER - RADIUS, CENTER - RADIUS
  return u - x, v - y, u + x, v + y


def arrow_segments(vector):
  """Return arrow segments representing last movement."""
  # body of the arrow
  x, y = 40 * vector
  u, v = CENTER - RADIUS + 10, CENTER - RADIUS + 10
  lines = [(u - x, v + y, u + x, v - y)]

  # head of the arrow
  c, s = np.cos(0.65), np.sin(0.65)
  rot_matrix = np.array(((c, -s), (s, c)))
  for mat in [rot_matrix, np.linalg.inv(rot_matrix)]:
    x1, y1 = 10 * np.dot(mat, vector)
    lines.append((u + x - x1, v - y + y1, u + x, v - y))

  return lines


def draw_text(draw, monster_speed, step, step_size, num_actions, position=None):
  """Draw informational text to image."""
  monster_text = f'MONSTER SPEED: {monster_speed}'
  step_text = f'STEP: {step}'

  actions_text = f'NUMBER OF ACTIONS: {num_actions}'
  size_text = f'STEP SIZE: {step_size}'
  draw.text((10, SIZE - 20), monster_text, (0, 0, 0))
  draw.text((10, SIZE - 40), actions_text, (0, 0, 0))
  draw.text((10, SIZE - 60), size_text, (0, 0, 0))
  draw.text((CENTER - 20, SIZE - 20), step_text, (0, 0, 0))
  if position is not None:
    radius_text = f'RADIUS: {np.linalg.norm(position):.3f}'
    draw.text((CENTER + 80, SIZE - 20), radius_text, (0, 0, 0))


def renderer(monster_angle,
             prev_monster_angle,
             position,
             prev_action_vector,
             result,
             reward,
             step,
             monster_speed,
             num_actions,
             step_size,
             is_caught,
             return_real=False):
  """Render an environment state as a PIL image."""

  c, s = np.cos(monster_angle), np.sin(monster_angle)
  rot_matrix = np.array(((c, -s), (s, c)))
  real_position = np.dot(rot_matrix, position)

  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)

  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=(0, 0, 0), width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=(0, 0, 0))
  draw_text(draw, monster_speed, step, step_size, num_actions, position)

  draw.ellipse(coords_to_rect(real_position), fill=RED)
  draw.ellipse(angle_to_rect(monster_angle), fill=(40, 200, 40))

  # drawing the arrow
  if prev_action_vector is not None:
    if is_caught:
      color = (255, 150, 0)
    else:
      color = (255, 255, 0)
    c, s = np.cos(prev_monster_angle), np.sin(prev_monster_angle)
    prev_rot_matrix = np.array(((c, -s), (s, c)))
    real_vector = np.dot(prev_rot_matrix, prev_action_vector)
    unit_vector = real_vector / np.linalg.norm(real_vector)
    lines = arrow_segments(unit_vector)
    for line in lines:
      draw.line(line, fill=color, width=4)

  # displaying the episode result
  if result is not None:
    white = (255,) * 3
    draw.text((CENTER - 10, CENTER + 30), result.upper(), white)
    draw.text((CENTER - 10, CENTER + 50), f'REWARD: {reward:.3f}', white)

  if return_real:
    return im, real_position
  return im


def render_agent_path(im, path):
  """Draw path onto im."""
  np_center = np.array((CENTER, CENTER))
  np_radius = np.array((RADIUS, -RADIUS))
  scaled_path = [tuple(np_center + np_radius * coord) for coord in path]
  draw = ImageDraw.Draw(im)
  draw.line(scaled_path, fill=RED, width=4)
  return im


def render_many_agents(positions, colors, step, step_size, num_actions, monster_speed):
  """Keep monster at (1, 0) and render agent positions."""
  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)
  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=(0, 0, 0), width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=(0, 0, 0))
  draw_text(draw, monster_speed, step, step_size, num_actions)
  draw.ellipse(angle_to_rect(0), fill=(40, 200, 40))  # monster themself

  for p, c in zip(positions, colors):
    draw.ellipse(coords_to_rect(p), fill=c)
  return im


def episode_as_video(py_env, policy, filename, tf_env=None):
  """Create mp4 video through py_environment render method."""
  print('Creating video with render method ...')

  if tf_env is None:
    tf_env = py_env

  fps = 10
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
  print(f'Video created and saved as {filename}')


def episode_as_gif(py_env, policy, filepath, fps=30, tf_env=None):
  """Create gif through py_environment render method."""

  if tf_env is None:
    tf_env = py_env

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
