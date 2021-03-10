import torch
from torchvision import transforms

# H36
ROOT = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/'
FILE_H36_TFR = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/source_data'
FILE_H36_IMG_ori = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/images/'
FILE_H36_IMG_aug = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/images_augmented/'
FILE_H36_POSE_3D = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/pose3d/'
FILE_H36_POSE_2D = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/pose2d/'
FILE_H36_POSE_INTR = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/intrinsics/'
FILE_H36_POSE_INTR_UNI = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/h36/intrinsics_univ/'
FILE_H36_IMG_TEST = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/test_data/images/'
FILE_H36_TFR_TEST = '/cluster/work/riner/users/PLR-2020/lechen/mp/test_tfr/'

# MP2
FILE_MP2_TFR = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/source_data'
FILE_MP2_IMG = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/mp2/images/'
FILE_MP2_POSE_2D = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/mp2/pose2d/'
FILE_MP2_VIS = '/cluster/work/riner/users/PLR-2020/lechen/mp/data_path/extracted_data/mp2/visibility/'

# Data Augmentation
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# tar -xf VOCtrainval_11-May-2012.tar
# ~2GB
VOC_DATASET = '/cluster/work/riner/users/PLR-2020/lechen/mp/VOC2012'

# model
MODEL_PATH = './model_parameter/'
HRNET = 'hrnet.pkl'
SEMGCN = 'semgcn.pkl'
FEEDFORWARD = 'feedforward.pkl'

# submission
SUBMISSION = 'submission.csv.gz'

# Sample parameter
NUM_SAMPLES_H36 = 636724
NUM_SAMPLES_TEST = 2181
NUM_SAMPLES_MPII = 18000
MIN_LEN = 9

# Training set
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 16
BATCHSIZE_TEST = 3
BATCHSIZE_VALI = 16
TRANSFORM = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For HRnet
GAUS_KERNEL = 3
GAUS_STD = 1
