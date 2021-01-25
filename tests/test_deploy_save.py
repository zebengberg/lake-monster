"""Test saved model from lake_monster.agent.deploy."""

import os
import random
import numpy as np
import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from lake_monster.environment.environment import LakeMonsterEnvironment
from lake_monster import configs, train
from lake_monster.agent import verify, deploy


def load_models():
  """Load existing saved models."""
  if os.path.exists(configs.AGENT_ID_PATH):

    deploy.save_model()
    agent = train.restore_existing_agent()
    model1 = verify.ModelWrapper(agent.q_net, agent.use_categorical)

    p = agent.env_params
    p['monster_speed'] = 3.5
    p['step_size'] = 0.02
    p['use_random_start'] = False

    env = LakeMonsterEnvironment(**p)
    step_increase = env.step_size / env.timeout_factor
    env = TFPyEnvironment(env)

    savepath = os.path.join(configs.TEMP_DIR, 'saved_model')
    model2 = tf.keras.models.load_model(savepath)

    return model1, model2, agent, env, step_increase

  return None


def update(state, action, step_increase):
  """Return new state."""

  # agent move
  step_size = state[0]
  monster_speed = state[1]
  action_vector = step_size * np.array([
      np.cos(2 * np.pi * action / 8),
      np.sin(2 * np.pi * action / 8)
  ])
  position = np.array([state[3], 0]) + action_vector
  r = np.linalg.norm(position)
  angle = state[4] - np.arctan2(position[1], position[0])

  # monster move
  arc_length = step_size * monster_speed
  s = np.sign(angle)
  if abs(angle) > arc_length:
    angle -= s * arc_length
  else:
    angle = 0

  # normalize
  if angle > np.pi:
    angle -= 2 * np.pi
  if angle < -np.pi:
    angle += 2 * np.pi
  return np.array([*state[:2], state[2] + step_increase, r, angle])


def test_model_agreement():
  """Ensure that the same policy accessed in different ways agree."""

  if (loaded := load_models()) is not None:
    model1, model2, _, _, _ = loaded

    for _ in range(65):
      state = [
          0.1 * random.random(),
          4 * random.random(),
          random.random(),
          random.random(),
          6.28 * random.random() - 3.14
      ]
      state = np.array(state)
      state = np.expand_dims(state, 0)

      p1 = model1.predict(state)
      p2 = model2.predict(state)
      assert (p1 == p2).all()
      print('.', end='', flush=True)
    print('\n\n')


def test_greedy():
  """Test that tf_agent greedy policy agrees with argmax decision."""

  if (loaded := load_models()) is not None:
    model, _, agent, env, _ = loaded

    for _ in range(10):
      timestep = env.reset()
      while not timestep.is_last():
        action = agent.agent.policy.action(timestep).action
        pred = model.predict(timestep.observation)[0]
        argmax = np.argmax(pred)
        assert action.numpy()[0] == argmax
        timestep = env.step(action)
      print('.', end='', flush=True)
    print('\n\n')


def run_with_prints():
  """Run episode and print details."""
  model, _, agent, env, step_increase = load_models()
  timestep = env.reset()
  while not timestep.is_last():

    print('#' * 65)
    state = timestep.observation.numpy()[0]
    print('old state:', state.round(3))

    action = agent.agent.policy.action(timestep).action
    pred = model.predict(timestep.observation)[0]
    argmax = np.argmax(pred)
    print('pred:', pred.round(3))
    print('action:', np.argmax(pred))
    timestep = env.step(action)

    # ensuring that homemade update agrees with tf_agent update
    assert action.numpy()[0] == argmax
    homemade_state = update(state, argmax, step_increase)
    tf_state = timestep.observation.numpy()[0]
    assert (abs(homemade_state - tf_state) < 1e-7).all()

    print('new state:', tf_state.round(3))
    input('Press any key to continue\n')
  print(timestep.reward.numpy())


if __name__ == '__main__':
  run_with_prints()
