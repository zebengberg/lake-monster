"""A script to remove all knowledge saved during agent training."""

import os
import shutil


if os.path.exists('videos/'):
  shutil.rmtree('videos/')

if os.path.exists('checkpoints/'):
  shutil.rmtree('checkpoints/')

if os.path.exists('logs/'):
  shutil.rmtree('logs/')

# clearing policies (for now!)
if os.path.exists('policies'):
  shutil.rmtree('policies')
