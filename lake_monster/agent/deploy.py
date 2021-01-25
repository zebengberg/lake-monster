"""Save model to deploy."""

import os
import uuid
import argparse
from lake_monster import configs, train, utils
from lake_monster.agent import verify, agent

deploy_params = {
    # environment params
    'n_actions': 8,
    'initial_step_size': 0.1,
    'min_step_size': 0.01,
    'timeout_factor': 3.0,
    'use_mini_rewards': True,
    'use_random_start': True,
    'use_random_monster_speed': True,

    # agent params
    'fc_layer_params': (100, 100),
    'dropout_layer_params': (0.1, 0.1),
    'learning_rate': 0.001,
    'epsilon_greedy': 0.1,
    'n_step_update': 6,
    'use_categorical': False,
    'use_step_schedule': True,
    'use_mastery': False,
    'use_evaluation': False
}


def run_train():
  """Run agent train for gh-pages deploy."""
  if os.path.exists(configs.AGENT_ID_PATH):
    a = train.restore_existing_agent()
  else:
    uid = str(uuid.uuid1().int)
    verify.verify_agent(uid, deploy_params)
    verify.log_graph(uid, deploy_params)
    utils.log_uid(uid)
    utils.log_params(uid, deploy_params)
    a = agent.Agent(uid, **deploy_params)

  train.launch_tb(a.get_uid())
  a.train_ad_infinitum()


def save_model():
  """Save currently saved model as tf_saved_model form."""
  # loading actively trained agent
  a = train.restore_existing_agent()
  model = verify.ModelWrapper(a.q_net, a.use_categorical)

  # making a single call so tf knows the input shape
  env = a.build_temp_env()
  x = env.reset().observation
  model(x)

  # saving as tf_saved_model
  savepath = os.path.join(configs.TEMP_DIR, 'saved_model')
  model.save(savepath)
  print(f'Model saved to {savepath}')
  print('#' * 65)
  print('\n\n')


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--save', action='store_true')


if __name__ == '__main__':
  args = parser.parse_args()
  if args.train:
    run_train()
  elif args.save:
    save_model()
