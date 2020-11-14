"""A tf-agent environment for the lake monster problem."""


import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from renderer import renderer


def rand_position():
  """Return a random point in the unit disk."""
  r = 0.01 * np.random.random()
  theta = np.random.random()
  x = r * np.cos(2 * np.pi * theta)
  y = r * np.sin(2 * np.pi * theta)
  return np.array((x, y), dtype=np.float32)


class LakeMonsterEnvironment(py_environment.PyEnvironment):
  """A tf-agent environment for the lake monster problem. In this environment,
  the monster remains fixed at the point (1, 0), and the player is confined to
  swim inside of the unit disk. After an action takes place, the player is
  rotated within the unit disk to simulate the monster traversing the lake
  shore. An observation of the environment consists of the monster's speed as
  well as the player's x, y, and r coordinates. Because of limits in tf's DQN
  agent, the action must be one-dimensional (instead of a possibly more natural
  dx, dy). See https://stats.stackexchange.com/questions/218407/ and
  https://github.com/tensorflow/agents/issues/97
  """

  def __init__(self, monster_speed=1.0,
               timeout_factor=3,
               step_size=0.1,
               num_actions=4,
               use_mini_rewards=False):
    super().__init__()

    self.monster_speed = round(monster_speed, 3)
    self.timeout_factor = timeout_factor
    self.step_size = round(step_size, 2)
    self.num_actions = num_actions
    self.use_mini_rewards = use_mini_rewards

    # total number of allowed steps
    self.duration = int(timeout_factor / step_size)

    # building the action_to_direction list

    def to_vector(action):
      return np.array((np.cos(action * 2 * np.pi / num_actions),
                       np.sin(action * 2 * np.pi / num_actions)))
    self.action_to_direction = [to_vector(a) for a in range(num_actions)]

    # building the rotation matrix
    self.monster_arc = self.step_size * self.monster_speed
    c, s = np.cos(self.monster_arc), np.sin(self.monster_arc)
    self.ccw_rot_matrix = np.array(((c, -s), (s, c)))
    self.cw_rot_matrix = np.array(((c, s), (-s, c)))

    self.num_steps = 0
    self.position = np.array((0.0, 0.0), dtype=np.float32)
    self.highest_r_attained = 0.0
    self.is_monster_caught_up = False

    self.position = rand_position()

    # TODO: use derived class for rendering?
    self.prev_monster_angle = 0.0  # only used in render method
    self.monster_angle = 0.0  # only used in render method
    self.prev_action_vector = None  # only used in render method

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1,
        name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=self._state.shape, dtype=np.float32, minimum=-10, maximum=10,
        name='observation')
    self._episode_ended = False

  @ property
  def _state(self):
    return np.array((self.step_proportion, self.monster_speed,
                     self.r, self.theta), dtype=np.float32)

  @ property
  def r(self):
    """Return the radius of the player."""
    return np.linalg.norm(self.position)

  @ property
  def step_proportion(self):
    """Return proportion of number of steps taken to number of steps allowed."""
    return self.num_steps / self.duration

  @ property
  def theta(self):
    """Return the angle of the player."""
    return np.arctan2(self.position[1], self.position[0])

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self.position = np.array((0.0, 0.0), dtype=np.float32)
    self.num_steps = 0
    self.highest_r_attained = 0.0
    self.is_monster_caught_up = False
    self.monster_angle = 0.0
    self.prev_action_vector = None
    return time_step.restart(self._state)

  def rotate(self):
    """Update the position after action applied."""
    self.prev_monster_angle = self.monster_angle

    # get agent off the x-axis to avoid boundary cases
    self.position += np.array((0, 1e-6 * (np.random.random() - 0.5)))

    y_sign = np.sign(self.position[1])
    if abs(y_sign) != 1.0:
      print('Agent position:', self.position)
      raise ValueError('The agent is on the x-axis, and it should not be!')

    if y_sign == 1.0:
      rot_matrix = self.cw_rot_matrix
    else:
      rot_matrix = self.ccw_rot_matrix

    rotated = np.dot(rot_matrix, self.position)
    if np.sign(rotated[1]) != y_sign:  # we've rotated past the x-axis
      self.monster_angle += np.arctan2(self.position[1], self.position[0])
      rotated = np.array((self.r, 0.0))
      self.is_monster_caught_up = True
    else:
      self.monster_angle += y_sign * self.monster_arc

    self.position = rotated
    if not self.is_monster_caught_up:
      self.highest_r_attained = max(self.highest_r_attained, self.r)

  def _step(self, action):
    if self._episode_ended:
      # previous action ended the episode so we ignore current action and reset
      return self.reset()

    action_vector = self.step_size * self.action_to_direction[action]
    self.prev_action_vector = action_vector
    prev_position = self.position
    self.position += action_vector
    # if
    # if self.position[1] == 0.0 and self.position[0] > 0:
    #   self.is_monster_caught_up = True
    self.rotate()
    self.num_steps += 1

    # made it out of the lake
    if self.r >= 1.0 or self.step_proportion >= 1:
      self._episode_ended = True
      reward, _ = self.determine_reward()
      return time_step.termination(self._state, reward=reward)

    # still swimming
    return time_step.transition(self._state, reward=0)

  def determine_reward(self):
    """If the episode has ended, return the reward and result."""
    if not self._episode_ended:
      raise ValueError('Episode has not ended, but determine_reward is called')

    if self.r >= 1.0:
      if round(self.position[1], 6) == 0.0 and self.position[0] > 0:
        return int(self.use_mini_rewards) * self.highest_r_attained, 'capture'
      return 1 + int(self.use_mini_rewards) * abs(self.theta), 'success'

    # slightly worse penalty for timeout than capture
    return int(self.use_mini_rewards) * self.highest_r_attained - 0.1, 'timeout'

  def render(self, mode='rgb_array'):
    # determine if episode just ended
    result = None
    reward = None
    if self._episode_ended:
      reward, result = self.determine_reward()
    params = {'monster_angle': self.monster_angle,
              'prev_monster_angle': self.prev_monster_angle,
              'position': self.position,
              'prev_action_vector': self.prev_action_vector,
              'result': result,
              'reward': reward,
              'step': self.num_steps,
              'monster_speed': self.monster_speed,
              'num_actions': self.num_actions,
              'step_size': self.step_size,
              'is_caught': self.is_monster_caught_up}
    im = renderer(**params)
    if mode == 'rgb_array':
      return np.array(im)
    im.show()
    return None
