"""Define classes derived from LakeMonsterEnvironment."""


from environment import LakeMonsterEnvironment
import numpy as np


class JumpingEnvironment(LakeMonsterEnvironment):
  """A LakeMonsterEnvironent in which the monster randomly teleports once per episode."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.jump_at_r = np.random.random()
    self.has_jumped = False

  def _reset(self):
    self.jump_at_r = np.random.random()
    self.has_jumped = False
    return super()._reset()

  def _step(self, action):
    if not self.has_jumped and self.r > self.jump_at_r:
      theta = 2 * np.pi * np.random.random()
      self.monster_angle += theta
      self.total_monster_rotation += theta
      self.has_jumped = True
    return super()._step(action)


class MultiMonsterEnvironment(LakeMonsterEnvironment):
  """A LakeMonsterEnvironment with many starting monsters equidistributed over the lake."""

  def __init__(self, n_monsters=2, **kwargs):

    self.n_monsters = n_monsters
    self.monster_angles = [2 * np.pi * i /
                           self.n_monsters for i in range(self.n_monsters)]
    self.total_monster_rotations = self.monster_angles.copy()
    super().__init__(**kwargs)

  def _reset(self):
    self.monster_angles = [2 * np.pi * i /
                           self.n_monsters for i in range(self.n_monsters)]
    self.total_monster_rotations = self.monster_angles.copy()
    return super()._reset()

  @property
  def _state(self):
    state = (self.step_size, self.monster_speed, self.step_proportion,
             self.r, *self.monster_angles)
    return np.array(state, dtype=np.float32)

  def _step(self, action):
    theta = self.move_agent(action)
    for i in range(self.n_monsters):
      self.monster_angles[i], self.total_monster_rotations[i] = self.rotate_monster(
          theta, self.monster_angles[i], self.total_monster_rotations[i])
    return self.conclude_step()

  def render(self, mode='rgb_array'):
    return super().render(mode=self.total_monster_rotations)
