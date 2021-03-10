import os
import config
import config.hrnet

from graph_utils import adj_mx_from_skeletion17

END_TO_END, SOFTARGMAX = (True, True)

# For Hrnet
INIT_WEIGHTS = True
TARGET_TYPE = config.hrnet.TARGET_TYPE
IMAGE0_SIZE = config.hrnet.IMAGE_SIZE
HEATMAP_SIZE = config.hrnet.HEATMAP_SIZE
SIGMA = config.hrnet.SIGMA
EXTRA = config.hrnet.EXTRA

# For SemGCN
EDGES = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5),
         (7, 0), (8, 7), (9, 8), (10, 9), (11, 8), (12, 11),
         (13, 12), (14, 8), (15, 14), (16, 15)]
NUM_JOINTS = 17
ADJ = adj_mx_from_skeletion17(num_joints=NUM_JOINTS,
                              edges=EDGES)
HID_DIM = 128
NUM_LAYERS = 4

# For END2END
LOSS_HEATMAP = 10
LOSS_POSE3D =10