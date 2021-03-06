import os
import subprocess
import config
import datetime

# HRNet Parameters
PRETRAINED = None
TARGET_TYPE = 'gaussian'
INIT_WEIGHTS = True
IMAGE_SIZE = [256, 256]
HEATMAP_SIZE = [64, 64]
NUM_JOINTS = 17
SIGMA = 2
EXTRA = {
    'PRETRAINED_LAYERS': [
        'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2',
        'transition2', 'stage3', 'transition3', 'stage4'
    ],
    'FINAL_CONV_KERNEL':
    1,
    'STAGE2': {
        'NUM_MODULES': 1,
        'NUM_BRANCHES': 2,
        'NUM_BLOCKS': [4, 4],
        'NUM_CHANNELS': [32, 64],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE3': {
        'NUM_MODULES': 4,
        'NUM_BRANCHES': 3,
        'NUM_BLOCKS': [4, 4, 4],
        'NUM_CHANNELS': [32, 64, 128],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE4': {
        'NUM_MODULES': 3,
        'NUM_BRANCHES': 4,
        'NUM_BLOCKS': [4, 4, 4, 4],
        'NUM_CHANNELS': [32, 64, 128, 256],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    }
}

USE_GPU = config.USE_GPU
GAUS_KERNEL = 3
GAUS_STD = 1
BASE_WEIGHTS = None