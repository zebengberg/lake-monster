"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import os
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from environment import LakeMonsterEnvironment
from renderer import episode_as_video
from stats import Stats

# suppressing some warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# hyperparameters
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
monster_speed = 1.3

# tf environments
train_py_env = LakeMonsterEnvironment(monster_speed)
eval_py_env = LakeMonsterEnvironment(monster_speed)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# agent and q network
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
global_step = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)

agent.initialize()
agent.train = common.function(agent.train)


# data pipeline
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,  # defaults to 1
    max_length=replay_buffer_max_length)

observers = [replay_buffer.add_batch]
driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers=observers)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,  # not sure!
    num_steps=2).prefetch(3)
iterator = iter(dataset)


# checkpoints and stats
CHECKPOINT_DIR = 'checkpoints/'
train_checkpointer = common.Checkpointer(
    ckpt_dir=CHECKPOINT_DIR,
    max_to_keep=3,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
train_checkpointer.initialize_or_restore()
stats = Stats()


def evaluate_agent(env, policy):
  """Use naive while loop to evaluate policy in single episode."""
  n_steps = 0

  ts = env.reset()
  n_steps = 0
  while not ts.is_last():
    action = policy.action(ts)
    ts = env.step(action.action)
    n_steps += 1

  reward = ts.reward.numpy()[0].item()  # item converts to native python type
  return reward, n_steps


def train_agent():
  """Train the agent until interrupted by user."""

  while True:
    driver.run()  # train a single episode
    experience, _ = next(iterator)
    agent.train(experience)

    train_step = agent.train_step_counter.numpy().item()
    if train_step % 10 == 0:
      print('|', end='', flush=True)
      reward, n_steps = evaluate_agent(eval_env, agent.policy)
      d = {'episode': train_step, 'reward': reward,
           'n_env_steps': n_steps, 'monster_speed': monster_speed}
      stats.add(d)

    if train_step % 500 == 0:
      print('')
      save_progress()
      print('Progress saved!')
      vid_file = f'videos/episode-{train_step}.mp4'
      episode_as_video(eval_py_env, agent.policy, vid_file, eval_env)


def save_progress():
  """Save train checkpoint and updated stats."""
  print('Saving progress to disk ...')
  train_checkpointer.save(global_step)
  stats.save()
  print('Progress saved.')


if __name__ == '__main__':
  train_agent()
