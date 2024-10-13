import sys
import os

dataflow_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 
sys.path.insert(0, dataflow_path)

import dataflow
from dataflow.config import init_config

cfg = init_config()
print(cfg['video'].keys())
scorer = dataflow.get_scorer('VideoMotionScorer', device='cuda')
score = scorer.evaluate({'video': 'test_video.mp4'})
print(f'The score of test_video.mp4 is {score}')