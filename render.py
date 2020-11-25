"""Utility functions for rendering an environment and displaying an episode in video."""


import numpy as np
from PIL import Image, ImageDraw


SIZE = 480
CENTER = SIZE // 2
RADIUS = 200
RED = (250, 50, 0)
BLACK = (0,) * 3
GREEN = (40, 200, 40)


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


def draw_text(draw, monster_speed, step, step_size, n_actions, r=None):
  """Draw informational text to image."""
  monster_text = f'MONSTER SPEED: {monster_speed:.3f}'
  step_text = f'STEP: {step}'
  actions_text = f'NUMBER OF ACTIONS: {n_actions}'
  size_text = f'STEP SIZE: {step_size:.3f}'

  draw.text((10, SIZE - 20), monster_text, BLACK)
  draw.text((10, SIZE - 40), actions_text, BLACK)
  draw.text((10, SIZE - 60), size_text, BLACK)
  draw.text((CENTER - 20, SIZE - 20), step_text, BLACK)

  if r is not None:
    radius_text = f'RADIUS: {r:.3f}'
    draw.text((CENTER + 80, SIZE - 20), radius_text, BLACK)


def renderer(r,
             total_agent_rotation,
             total_monster_rotation,
             action_vector,
             result,
             reward,
             step,
             monster_speed,
             n_actions,
             step_size,
             is_caught,
             return_real=False,
             multi_monster_rotations=None):
  """Render an environment state as a PIL image."""

  c, s = np.cos(total_agent_rotation), np.sin(total_agent_rotation)
  agent_rot_matrix = np.array(((c, -s), (s, c)))
  agent_position = np.dot(agent_rot_matrix, (r, 0))

  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)

  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=BLACK, width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=BLACK)
  draw_text(draw, monster_speed, step, step_size, n_actions, r)

  draw.ellipse(coords_to_rect(agent_position), fill=RED)
  if multi_monster_rotations is None:
    multi_monster_rotations = [total_monster_rotation]
  for monster in multi_monster_rotations:
    draw.ellipse(angle_to_rect(monster), fill=GREEN)

  # drawing the arrow
  if action_vector is not None:
    if is_caught:
      color = (255, 150, 0)
    else:
      color = (255, 255, 0)

    real_vector = np.dot(agent_rot_matrix, action_vector)
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
    return im, agent_position
  return im


def render_agent_path(im, path):
  """Draw path onto im."""
  np_center = np.array((CENTER, CENTER))
  np_radius = np.array((RADIUS, -RADIUS))
  scaled_path = [tuple(np_center + np_radius * coord) for coord in path]
  draw = ImageDraw.Draw(im)
  draw.line(scaled_path, fill=RED, width=4)
  return im


def render_many_agents(positions, colors, step, step_size, n_actions, monster_speed):
  """Keep monster at (1, 0) and render agent positions."""
  im = Image.new('RGB', (480, 480), (237, 201, 175))
  draw = ImageDraw.Draw(im)
  draw.ellipse((CENTER - RADIUS,) * 2 + (CENTER + RADIUS,) * 2,
               fill=(0, 0, 255), outline=BLACK, width=4)
  draw.ellipse((CENTER - 2,) * 2 + (CENTER + 2,) * 2, fill=BLACK)
  draw_text(draw, monster_speed, step, step_size, n_actions)
  draw.ellipse(angle_to_rect(0), fill=GREEN)  # monster themself

  for p, c in zip(positions, colors):
    draw.ellipse(coords_to_rect(p), fill=c)
  return im
