"""A script for training the RL agent."""


from agent import Agent
from test_agent import test_agent, log_graph

params = {'num_actions': 8,
          'step_size': 0.05,
          'initial_monster_speed': 1.0,
          'timeout_factor': 3,
          'fc_layer_params': (80, 80),
          'learning_rate': 0.01,
          'epsilon_greedy': 0.2,
          'penalty_per_step': 0.1}


if __name__ == '__main__':
  test_agent(params)
  log_graph(params)
  a = Agent(**params)
  a.train_ad_infinitum()
