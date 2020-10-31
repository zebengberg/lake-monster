"""A script to remove all knowledge saved during agent training."""

import os
import shutil
from stats import FILE_NAME as stats_file


if os.path.exists(stats_file):
  os.remove(stats_file)

if os.path.exists('videos/'):
  shutil.rmtree('videos/')

if os.path.exists('checkpoints/'):
  shutil.rmtree('checkpoints/')
