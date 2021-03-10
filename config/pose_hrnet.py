import os
import config
import config.hrnet

USE_GPU = config.USE_GPU
NUM_JOINTS = config.NUM_JOINTS
SOFTARGMAX = True



# HRNet Parameters
INIT_WEIGHTS = True
TARGET_TYPE = config.hrnet.TARGET_TYPE
IMAGE0_SIZE = config.hrnet.IMAGE_SIZE
HEATMAP_SIZE = config.hrnet.HEATMAP_SIZE
SIGMA = config.hrnet.SIGMA
EXTRA = config.hrnet.EXTRA

# Parameters for generating heatmaps from coordinates
GAUS_KERNEL = 3
GAUS_STD = 1
