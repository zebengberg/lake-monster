"""A script for training the RL agent."""

import webbrowser
import tensorboard
from agent import Agent
from test_agent import test_agent, log_graph


def launch_tb():
  """Launch tensorboard in a new browswer tab."""
  print('Launching tensorboard. It will open shortly in a browser tab ...')
  tb = tensorboard.program.TensorBoard()
  tb.configure(logdir='logs')
  url = tb.launch()
  webbrowser.open_new_tab(url)


params = {'num_actions': 8,
          'initial_step_size': 0.1,
          'initial_monster_speed': 1.0,
          'timeout_factor': 3,
          'fc_layer_params': (20, 20),
          'learning_rate': 0.01,
          'epsilon_greedy': 0.2,
          'penalty_per_step': 0.0,
          'name': 'alice'}


if __name__ == '__main__':
  launch_tb()
  test_agent(params)
  log_graph(params)
  a = Agent(**params)
  a.train_ad_infinitum()
