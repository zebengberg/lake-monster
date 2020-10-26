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


def renderer(monster_angle, position, prev_action_vector):
  """Render an environment state as a PIL image."""

  c, s = np.cos(monster_angle), np.sin(monster_angle)
  rot_matrix = np.array(((c, -s), (s, c)))
  real_position = np.dot(rot_matrix, position)
  real_vector = np.dot(rot_matrix, prev_action_vector)
  real_vector = real_vector / np.linalg.norm(real_vector)

  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)

  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=(0, 0, 0), width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=(0, 0, 0))

  draw.rectangle(coords_to_rect(real_position), fill=(250, 50, 0))
  draw.rectangle(angle_to_rect(monster_angle), fill=(40, 200, 40))
  draw.line(vector_to_rect(real_vector), fill=(255, 255, 0), width=4)

  return im


# TODO: filepath arg, policy arg
def episode_as_video(env):
  """Create py environment video through render method."""
  print('Creating video from render method ...')

  with imageio.get_writer('test_vid.mp4', fps=30) as video:
    time_step = env.reset()
    while not time_step.is_last():
      action = np.random.uniform(low=0, high=2*np.pi)
      time_step = env.step(action)
      video.append_data(env.render())
    print(time_step.reward)
