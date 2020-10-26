"""A tf-agent policy, driver, replay_buffer, and agent for the monster lake problem."""

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.trajectories import trajectory
from environment import LakeMonsterEnvironment
from renderer import episode_as_video


# hyperparameters
num_iterations = 50000
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
num_eval_episodes = 10
eval_interval = 1000
log_interval = 200
collect_steps_per_iteration = 1

# tf environments
train_py_env = LakeMonsterEnvironment()
eval_py_env = LakeMonsterEnvironment()
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


episode_as_video(eval_py_env, agent.policy, 'videos/agent.mp4', eval_env)


# metrics
def print_metrics(env, policy, num_episodes=100):
  """Handmade metric to compute return and steps for eval_env."""

  total_return = 0.0
  total_steps = 0
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = env.step(action_step.action)
      total_steps += 1
    total_return += time_step.reward.numpy()[0]

  avg_return = total_return / num_episodes

  print('avg_return =', avg_return)
  print('avg_steps =', total_steps / num_episodes)
  return avg_return


x = 1/0


# data collection
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2, single_deterministic_pass=False).prefetch(3)
iterator = iter(dataset)


# without drivers
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)


# need some initial collection
collect_data(train_env, agent.policy, replay_buffer, 100)


# Evaluate the agent's policy once before training.
avg_return = print_metrics(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy,
               replay_buffer, collect_steps_per_iteration)
  # Sample a batch of data from the buffer and update the agent's network.

  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss
  step = agent.train_step_counter.numpy()
  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))
  if step % eval_interval == 0:
    avg_return = print_metrics(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    episode_as_video(eval_py_env, agent.policy, f'videos/{step}.mp4')


# num_episodes = tf_metrics.NumberOfEpisodes()
# env_steps = tf_metrics.EnvironmentSteps()
# observers = [replay_buffer.add_batch, num_episodes, env_steps]
# driver = dynamic_episode_driver.DynamicEpisodeDriver(
#     train_env,
#     agent.collect_policy,
#     observers=observers,
#     num_episodes=eval_interval)
# final_time_step, policy_state = driver.run()
# dataset = replay_buffer.as_dataset(
#     num_parallel_calls=3, sample_batch_size=batch_size,
#     num_steps=2).prefetch(3)
# iterator = iter(dataset)
# for _ in range(num_iterations):
#   final_time_step, policy_state = driver.run(final_time_step, policy_state)
#   print(final_time_step.reward.numpy())
#   print(final_time_step.observation.numpy())
#   print(env_steps.result().numpy())
#   print(num_episodes.result().numpy())
#   episode_as_video(eval_py_env, agent.policy, 'agent.mp4')
