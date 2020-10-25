"""A tf-agent environment and tests for the lake monster problem."""

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
    self.monster_speed = 2.0
    self.step_size = 1e-2
    self.num_steps = 0
    self.position = np.array([0, 0], dtype=np.float32)

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(4,), dtype=np.float32, name='observation')
    self._episode_ended = False

  @property
  def _state(self):
    return np.array([self.monster_speed, self.x, self.y, self.r])

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
    self.position = np.array([0, 0], dtype=np.float32)
    self.num_steps = 0
    return ts.restart(np.array([self._state], dtype=np.float32))

  def rotate(self):
    """Update the position to reflect monster movement."""

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    # scaling action to step_size
    action /= np.linalg.norm(action)
    action *= self.step_size
    self.position += action
    if self.r > 1.0:
      self._episode_ended = True

    # forcing episode to close if taken over 1000 steps
    self.num_steps += 1
    if self.num_steps >= 1000:
      self._episode_ended = True

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
