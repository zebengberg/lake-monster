"""Utility functions for rendering an environment and displaying an episode in video."""

from PIL import Image, ImageDraw
import imageio
import numpy as np


SIZE = 480
CENTER = SIZE // 2
RADIUS = 200


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
  lines = [(u - x, v - y, u + x, v + y)]

  # head of the arrow
  c, s = np.cos(0.65), np.sin(0.65)
  rot_matrix = np.array(((c, -s), (s, c)))
  for mat in [rot_matrix, np.linalg.inv(rot_matrix)]:
    x1, y1 = 10 * np.dot(mat, vector)
    lines.append((u + x - x1, v + y - y1, u + x, v + y))

  return lines


def renderer(monster_angle, position, prev_action_vector, result, monster_speed, step):
  """Render an environment state as a PIL image."""

  c, s = np.cos(monster_angle), np.sin(monster_angle)
  rot_matrix = np.array(((c, -s), (s, c)))
  real_position = np.dot(rot_matrix, position)

  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)

  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=(0, 0, 0), width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=(0, 0, 0))

  draw.rectangle(coords_to_rect(real_position), fill=(250, 50, 0))
  draw.rectangle(angle_to_rect(monster_angle), fill=(40, 200, 40))

  monster_text = f'MONSTER SPEED: {monster_speed}'
  step_text = f'STEP: {step}'
  draw.text((CENTER - 150, SIZE - 20), monster_text, (0, 0, 0))
  draw.text((CENTER + 50, SIZE - 20), step_text, (0, 0, 0))

  if prev_action_vector is not None:
    real_vector = np.dot(rot_matrix, prev_action_vector)
    unit_vector = real_vector / np.linalg.norm(real_vector)
    lines = arrow_segments(unit_vector)
    for line in lines:
      draw.line(line, fill=(255, 255, 0), width=4)

  if result is not None:
    draw.text((CENTER - 10, CENTER + 30), result.upper(), (255, 255, 255))

  return im


def episode_as_video(py_env, policy, filename, tf_env=None):
  """Create py environment video through render method."""
  print('Creating video from render method ...')

  if tf_env is None:
    tf_env = py_env

  fps = 20
  with imageio.get_writer(filename, fps=fps) as video:
    time_step = tf_env.reset()
    video.append_data(py_env.render())
    while not time_step.is_last():
      action = policy.action(time_step).action
      time_step = tf_env.step(action)
      video.append_data(py_env.render())
    for _ in range(3 * fps):  # play for 3 more seconds
      video.append_data(py_env.render())
