"""A script for training the RL agent."""

from agent import Agent

params = {'num_actions': 6,
          'step_size': 0.05,
          'initial_monster_speed': 1.0,
          'timeout_factor': 3,
          'fc_layer_params': (200,),
          'learning_rate': 1e-1,
          'epsilon_greedy': 0.2}

# create a grid search function

a = Agent(**params)
a.train_ad_infinitum()
