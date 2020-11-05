"""A script to remove all knowledge saved during agent training."""

import os
import shutil


if os.path.exists('videos/'):
  shutil.rmtree('videos/')

if os.path.exists('checkpoints/'):
  shutil.rmtree('checkpoints/')

if os.path.exists('logs/'):
  shutil.rmtree('logs/')

if os.path.exists('agent_id.txt'):
  os.remove('agent_id.txt')


# uncomment lines below to clean out everything

# if os.path.exists('policies/'):
#   shutil.rmtree('policies/')
# if os.path.exists('results.json'):
#   os.remove('results.json')
