"""Portable path constants for the zscd package.

All filesystem locations derive from the repository location (this file's
directory) or environment overrides, so the same code runs on any machine:

  ZSCD_CKPT_DIR   override checkpoint directory (default: <repo>/checkpoints)
  ZSCD_DATA_ROOT  override test2 dataset root  (default: ~/Dataset/test2)
"""
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_DIR, 'results')
CKPT_DIR = os.environ.get('ZSCD_CKPT_DIR', os.path.join(REPO_DIR, 'checkpoints'))
DATA_ROOT = os.environ.get('ZSCD_DATA_ROOT', os.path.expanduser('~/Dataset/test2'))
OPENAI_CLIP_CACHE = os.path.expanduser('~/.cache/clip')
