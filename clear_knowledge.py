"""Remove checkpoints, stats, and videos creating during agent training."""

import os
import glob
from stats import FILE_NAME as stats_file
from test_environment import TEST_VIDEO_FILENAME as test_video_file

if os.path.exists(test_video_file):
  os.remove(test_video_file)

if os.path.exists(stats_file):
  os.remove(stats_file)

videos = glob.glob('videos/*')
for v in videos:
  os.remove(v)

checkpoints = glob.glob('checkpoints/*')
for c in checkpoints:
  os.remove(c)
