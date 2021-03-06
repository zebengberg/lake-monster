"""A tf-agent environment for the lake monster problem."""

import random
from dataclasses import dataclass
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step
from lake_monster.environment.render import renderer


@dataclass
class LakeMonsterEnvironment(PyEnvironment):
  """A tf-agent environment for the lake monster problem. In this environment,
  the agent remains fixed on the positive x-axis. After each action taken by the
  agent, both the agent and the monster are rotated so that the agent is
  returned to the x-axis."""

  monster_speed: float = 1.0
  timeout_factor: float = 3.0
  step_size: float = 0.1
  n_actions: int = 8
  use_mini_rewards: bool = True
  use_random_start: bool = False
  use_random_monster_speed: bool = False

  def __post_init__(self):
    super().__init__()

    # building the action_to_direction list
    def to_vector(action):
      return np.array((np.cos(action * 2 * np.pi / self.n_actions),
                       np.sin(action * 2 * np.pi / self.n_actions)))
    self.action_to_direction = [to_vector(a) for a in range(self.n_actions)]

    # state variables
    self.r = 0.0
    self.monster_angle = 0.0
    self.n_steps = 0

    # for rewards
    self.highest_r_attained = 0.0
    self.is_monster_caught_up = False

    # for rendering
    self.prev_agent_rotation = 0.0
    self.total_agent_rotation = 0.0
    self.total_monster_rotation = 0.0
    self.action_vector = None

    self._action_spec = BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=self.n_actions - 1,
        name='action'
    )
    self._observation_spec = BoundedArraySpec(
        shape=self._state.shape,
        dtype=np.float32,
        minimum=-10,
        maximum=10,
        name='observation'
    )
    self._episode_ended = False

  def _reset(self):
    self._episode_ended = False

    if self.use_random_start:
      # sometimes start near the origin, sometimes completely random
      self.monster_angle = 2 * np.pi * (random.random() - 0.5)
      if random.random() > 0.5:
        self.r = random.random()
      else:
        self.r = random.random() / 2000

    else:
      self.r = 0.0
      self.monster_angle = 0.0

    if self.use_random_monster_speed:  # monster speed between 2.5 and 4.5
      self.monster_speed = 2.5 + 2.0 * random.random()

    self.n_steps = 0
    self.highest_r_attained = 0.0
    self.is_monster_caught_up = False
    self.prev_agent_rotation = 0.0
    self.total_agent_rotation = 0.0
    self.total_monster_rotation = self.monster_angle
    self.action_vector = None
    return time_step.restart(self._state)

  @property
  def _state(self):
    return np.array((
        self.step_size,
        self.monster_speed,
        self.step_proportion,
        self.r,
        self.monster_angle
    ), dtype=np.float32)

  @property
  def step_proportion(self):
    """Return proportion of number of steps taken to number of steps allowed."""
    return self.n_steps * self.step_size / self.timeout_factor

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_info(self):
    print('Not implemented')

  def get_state(self):
    return self._state

  def set_state(self, state):
    print('Not implemented')

  def rotate_monster(self, theta, monster_angle, total_rotation):
    """Helper function for _step method."""

    # agent movement
    monster_angle -= theta
    if monster_angle > np.pi:
      monster_angle -= 2 * np.pi
    elif monster_angle < -np.pi:
      monster_angle += 2 * np.pi

    # monster movement
    arc_length = self.step_size * self.monster_speed
    if abs(monster_angle) > arc_length:
      s = np.sign(monster_angle)
      monster_angle -= s * arc_length
      total_rotation -= s * arc_length
    else:
      total_rotation -= monster_angle
      monster_angle = 0.0
      self.is_monster_caught_up = True

    return monster_angle, total_rotation

  def move_agent(self, action):
    """Helper function for _step method."""
    self.action_vector = self.step_size * self.action_to_direction[action]
    position = np.array((self.r, 0)) + self.action_vector
    if position[0] == 0.0:
      position[0] += random.random() * 1e-6  # avoid arctan issues

    self.r = np.linalg.norm(position)
    theta = np.arctan2(position[1], position[0])
    self.prev_agent_rotation = self.total_agent_rotation
    self.total_agent_rotation += theta
    return theta

  def conclude_step(self):
    """Helper function for _step method."""
    if self._episode_ended:
      return self.reset()

    if self.r < 1 / self.monster_speed:  # allowing "reset"
      self.is_monster_caught_up = False

    if not self.is_monster_caught_up:
      self.highest_r_attained = max(self.highest_r_attained, self.r)

    self.n_steps += 1

    # made it out of the lake
    if self.r >= 1.0 or self.step_proportion >= 1:
      self._episode_ended = True
      reward, _ = self.determine_reward()
      return time_step.termination(self._state, reward=reward)

    # still swimming
    return time_step.transition(self._state, reward=0)

  def _step(self, action):
    theta = self.move_agent(action)
    self.monster_angle, self.total_monster_rotation = self.rotate_monster(
        theta, self.monster_angle, self.total_monster_rotation)
    return self.conclude_step()

  def determine_reward(self):
    """If the episode has ended, return the reward and result."""
    if not self._episode_ended:
      raise ValueError('Episode has not ended, but determine_reward is called')

    if self.r >= 1.0:
      if self.monster_angle == 0.0:
        return int(self.use_mini_rewards) * self.highest_r_attained, 'capture'
      return 1 + int(self.use_mini_rewards) * abs(self.monster_angle), 'success'

    # slightly worse penalty for timeout than capture
    return int(self.use_mini_rewards) * self.highest_r_attained - 0.1, 'timeout'

  def render(self, mode='rgb_array'):
    # determine if episode just ended
    result = None
    reward = None
    if self._episode_ended:
      reward, result = self.determine_reward()
    params = {'r': self.r,
              'prev_agent_rotation': self.prev_agent_rotation,
              'total_agent_rotation': self.total_agent_rotation,
              'total_monster_rotation': self.total_monster_rotation,
              'action_vector': self.action_vector,
              'result': result,
              'reward': reward,
              'step': self.n_steps,
              'monster_speed': self.monster_speed,
              'n_actions': self.n_actions,
              'step_size': self.step_size,
              'is_caught': self.is_monster_caught_up}

    # monkey patch for MultiMonsterEnvironment to avoid changing ABC signature
    if isinstance(mode, list):
      if mode[-1] == 'return_real':
        params['return_real'] = True
        params['multi_monster_rotations'] = mode[:-1]
        return renderer(**params)
      params['multi_monster_rotations'] = mode
      im = renderer(**params)
      return np.array(im)

    if mode == 'return_real':
      params['return_real'] = True
      return renderer(**params)

    im = renderer(**params)
    if mode == 'rgb_array':
      return np.array(im)
    im.show()
    return None
