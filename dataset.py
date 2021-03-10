import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from configuration import *
from config.hrnet import GAUS_KERNEL, GAUS_STD


class h36(Dataset):
    def __init__(self,
                 list,
                 length,
                 transform,
                 pose3d_mean,
                 pose3d_std,
                 num_joints=17):
        self.list = list
        self.length = length
        self.transform = transform
        self.pose3d_mean = pose3d_mean
        self.pose3d_std = pose3d_std
        self.gaussian_filter = GaussianSmoothing2D(num_joints, GAUS_KERNEL,
                                                   GAUS_STD)

    def __len__(self):
        return self.length

    def __getitem__(self, item):

        # Read raw data
        img = Image.open(FILE_H36_IMG_aug + self.list[item] + '.jpg')
        pose3d = np.loadtxt(FILE_H36_POSE_3D + self.list[item] + '.txt')
        pose2d = np.loadtxt(FILE_H36_POSE_2D + self.list[item] + '.txt')

        # Normalize pose3d (reference: util.py in project2)
        pelvis = np.repeat(pose3d[0, :].reshape([-1, 3]), 17, axis=0)
        pose3d_norm = (pose3d - pelvis - self.pose3d_mean) / self.pose3d_std

        # Transform to tensor
        pose2d = torch.from_numpy(pose2d)

        # Data augmentation
        np.random.seed()

        # Normalization
        img = self.transform(img)

        # remove out of index error
        joints = pose2d
        joints = joints.to(dtype=torch.int16)
        map_dim = 64
        downscale = int(256 / map_dim)
        x, y = joints[:, 0].long(), joints[:, 1].long()
        idx = item

        while (max(x) // downscale >= 64) or (max(y) // downscale >= 64):
            # item = item + 1
            # Read raw data
            idx += 1
            img = Image.open(FILE_H36_IMG_aug + self.list[idx] + '.jpg')
            pose3d = np.loadtxt(FILE_H36_POSE_3D + self.list[idx] + '.txt')
            pose2d = np.loadtxt(FILE_H36_POSE_2D + self.list[idx] + '.txt')

            # Normalize pose3d (reference: util.py in project2)
            pelvis = np.repeat(pose3d[0, :].reshape([-1, 3]), 17, axis=0)
            pose3d_norm = (pose3d - pelvis - self.pose3d_mean) / self.pose3d_std

            # Transform to tensor
            pose2d = torch.from_numpy(pose2d)

            # Data augmentation
            np.random.seed()

            # Normalization
            img = self.transform(img)

            joints = pose2d
            joints = joints.to(dtype=torch.int16)
            map_dim = 64
            downscale = int(256 / map_dim)
            x, y = joints[:, 0].long(), joints[:, 1].long()

        # Get heatmap
        heatmap = self._generate_2Dheatmaps(pose2d)

        # Flattern
        pose2d, pose3d, pose3d_norm = pose2d.flatten(), pose3d.flatten(), pose3d_norm.flatten()

        return img, pose3d, pose3d_norm, pose2d, heatmap

    # Adapted from: https://github.com/meetvora/PoseNet/blob/master/core/data.py
    def _generate_2Dheatmaps(self, joints: torch.Tensor,
                             map_dim: int = 64) -> torch.Tensor:
        """ Generates 2d heatmaps from coordinates.
        Arguments:
            joints (torch.Tensor): An individual target tensor of shape (num_joints, 2).
        Returns:
            maps (torch.Tensor): 3D Tensor with gaussian activation at joint locations (num_joints, map_dim, map_dim)
        """
        # joints = torch.round(joints)
        # joints = torch.clamp(joints, min=0, max=255)
        joints = joints.to(dtype=torch.int16)
        num_joints = joints.shape[0]
        # print(num_joints)
        downscale = int(256 / map_dim)
        maps = torch.zeros((num_joints, map_dim, map_dim))
        x, y = joints[:, 0].long(), joints[:, 1].long()
        maps[np.arange(num_joints), x // downscale, y // downscale] = 1
        maps = self.gaussian_filter(maps)
        return maps


class h36_test(Dataset):
    def __init__(self,
                 list,
                 transfrom
                 ):
        self.list = list
        self.transform =transfrom

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        img = Image.open(FILE_H36_IMG_TEST + self.list[item])
        img = self.transform(img)

        return img

# Adapted from: https://github.com/meetvora/PoseNet/blob/master/core/data.py
class GaussianSmoothing2D(torch.nn.Module):
    """
	Arguments:
		channels (int): Number of channels of input. Output will have same number of channels.
		kernel_size (int): Size of the gaussian kernel.
		sigma (float): Standard deviation of the gaussian kernel.
		dim (int): Number of dimensions of the data.
		input_size (int): (H, W) Dimension of channel. Assumes H = W.
	"""

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 sigma: float,
                 dim: int = 2,
                 input_size: int = 64) -> None:
        super(GaussianSmoothing2D, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
              torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.num_channels = channels
        self.dim_input = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
		Apply gaussian filter to input.
		Arguments:
			input (torch.Tensor): Input of shape (C, H, W) to apply gaussian filter on.
		Returns:
			filtered (torch.Tensor): Filtered output of same shape.
		"""
        x = F.pad(x, (1, 1, 1, 1))
        x = x.unsqueeze(0).float()
        x = F.conv2d(x, weight=self.weight, groups=self.groups).squeeze()
        channel_norm = torch.norm(x.view(self.num_channels, -1), 2, 1)
        channel_norm = channel_norm.view(-1, 1).repeat(
            1, self.dim_input * self.dim_input)
        channel_norm = channel_norm.view(self.num_channels, self.dim_input,
                                         self.dim_input)
        return (x / channel_norm)

