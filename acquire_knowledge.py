"""A script for training the RL agent."""


from agent import Agent

params = {'num_actions': 8,
          'step_size': 0.2,
          'initial_monster_speed': 1.0,
          'timeout_factor': 3,
          'fc_layer_params': (50, 50),
          'learning_rate': 0.01,
          'epsilon_greedy': 0.2,
          'penalty_per_step': 0.01}


a = Agent(**params)
a.train_ad_infinitum()
