"""Save model to deploy."""

import os
import uuid
from lake_monster import configs, train, utils
from lake_monster.agent import verify, agent


def run_train():
  """Run agent train for gh-pages deploy."""
  uid = str(uuid.uuid1().int)

  params = {
      # environment params
      'n_actions': 8,
      'initial_step_size': 0.1,
      'initial_monster_speed': 3.5,
      'timeout_factor': 3.0,
      'use_mini_rewards': True,

      # agent params
      'fc_layer_params': (100, 100),
      'dropout_layer_params': (0.5, 0.5),
      'learning_rate': 0.001,
      'epsilon_greedy': 0.1,
      'n_step_update': 5,
      'use_categorical': False,
      'use_step_schedule': True,
      'use_mastery': True,
  }

  verify.verify_agent(uid, params)
  verify.log_graph(uid, params)
  utils.log_uid(uid)
  utils.log_params(uid, params)
  a = agent.Agent(uid, **params)
  train.launch_tb(a.get_uid())
  a.train_ad_infinitum()


def save_model():
  """Save currently saved model as tf_saved_model form."""
  # loading actively trained agent
  a = train.restore_existing_agent()
  model = verify.ModelWrapper(a.q_net.copy(), a.use_categorical)

  # making a single call so tf knows the input shape
  env = a.build_temp_env()
  x = env.reset().observation
  model(x)

  # saving as tf_saved_model
  savepath = os.path.join(configs.TEMP_DIR, 'saved_model')
  model.save(savepath)


if __name__ == '__main__':
  save_model()
