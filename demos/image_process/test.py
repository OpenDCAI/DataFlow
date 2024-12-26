# run "python demos/image_dedup/test.py --config configs/process/image_deduplicate.yaml" under main folder

import sys
sys.path.insert(0, '../DataFlow')

from dataflow.utils.utils import process

if __name__ == '__main__':
    process()
