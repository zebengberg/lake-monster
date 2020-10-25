"""A tf-agent environment for the lake monster problem."""


import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class LakeMonsterEnvironment(py_environment.PyEnvironment):
  """A tf-agent environment for the lake monster problem. In this environment,
  the monster remains fixed at the point (1, 0), and the player is confined to
  swim inside of the unit disk. After an action takes place, the player is
  rotated within the unit disk to simulate the monster traversing the lake
  shore. An observation of the environment consists of the monster's speed as
  well as the player's x, y, and r coordinates."""

  def __init__(self):
    super().__init__()

    self.monster_speed = 0.75
    self.step_size = 0.035

    # building the rotation matrix
    theta = self.step_size * self.monster_speed
    c, s = np.cos(theta), np.sin(theta)
    self.ccw_rot_matrix = np.array(((c, -s), (s, c)))
    self.cw_rot_matrix = np.array(((c, s), (-s, c)))

    self.num_steps = 0
    self.max_steps = 1000
    self.position = np.array((0.0, 0.0), dtype=np.float32)

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(5,), dtype=np.float32, minimum=-10, maximum=10, name='observation')
    self._episode_ended = False

  @property
  def _state(self):
    return np.array((self.num_steps / self.max_steps, self.monster_speed,
                     *self.position, self.r), dtype=np.float32)

  @property
  def r(self):
    """Return the radius of the player."""
    return np.linalg.norm(self.position)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self.position = np.array((0.0, 0.0), dtype=np.float32)
    self.num_steps = 0
    return ts.restart(self._state)

  def rotate(self):
    """Update the position to reflect monster movement."""
    if self.position[1] > 0:
      rotated = np.dot(self.cw_rot_matrix, self.position)
    elif self.position[1] < 0:
      rotated = np.dot(self.ccw_rot_matrix, self.position)
    else:
      rotated = self.position

    # in case we rotate past the positive y-axis, we need to adjust
    if np.sign(self.position[1]) != np.sign(rotated[1]):
      self.position = np.array((self.r, 0.0))
    else:
      self.position = rotated

  def _step(self, action):
    if self._episode_ended:
      # previous action ended the episode so we ignore current action and reset
      return self.reset()

    # scaling action to step_size
    action = action / np.linalg.norm(action)
    action *= self.step_size
    self.position += action
    self.rotate()
    self.num_steps += 1

    # forcing episode to end if taking too long
    if self.num_steps >= self.max_steps:
      self._episode_ended = True
      return ts.termination(self._state, reward=-1)

    # made it out of the lake
    if self.r > 1.0:
      self._episode_ended = True
      if self.position[1] == 0.0 and self.position[0] > 0:
        reward = -1
      else:
        reward = 1
      return ts.termination(self._state, reward=reward)

    # still swimming
    return ts.transition(self._state, reward=0)

  def render(self):
    pass
