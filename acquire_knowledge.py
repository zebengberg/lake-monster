"""A script for training the RL agent."""

from agent import Agent

params = {'num_actions': 8,
          'step_size': 0.05,
          'initial_monster_speed': 2.0,
          'monster_speed_step': 0.05,
          'hidden_layer_nodes': 100}

a = Agent(**params)
a.train_ad_infinitum()
