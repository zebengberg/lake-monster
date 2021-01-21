"""Define common data paths used in modules."""

import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
POLICY_DIR = os.path.join(DATA_DIR, 'policies')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
JS_MODEL_DIR = os.path.join(DATA_DIR, 'js_model')
AGENT_ID_PATH = os.path.join(DATA_DIR, 'agent_id.txt')
RESULTS_PATH = os.path.join(DATA_DIR, 'results.json')
BACKUP_RESULTS_PATH = os.path.join(DATA_DIR, 'backup.json')

if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)
if not os.path.exists(VIDEO_DIR):
  os.mkdir(VIDEO_DIR)
if not os.path.exists(ASSETS_DIR):
  os.mkdir(ASSETS_DIR)
if not os.path.exists(TEMP_DIR):
  os.mkdir(TEMP_DIR)
