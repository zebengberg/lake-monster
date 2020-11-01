"""A script for training the RL agent."""

from agent import Agent

params = {'num_actions': 6,
          'step_size': 0.05,
          'initial_monster_speed': 1.5,
          'timeout_factor': 3,
          'hidden_layer_nodes': 100}

a = Agent(**params)
a.train_ad_infinitum()
