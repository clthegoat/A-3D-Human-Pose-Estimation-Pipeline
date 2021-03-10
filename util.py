import numpy as np
import torch.nn as nn
import torch
from configuration import *

def normalize(pose3d, mean, std, num_joints):
    pelvis = np.repeat(pose3d[0, :].reshape([-1, 3]), 17, axis=0)
    # Before normalization and un-normalization, pelvis coordniates need to be substrcated from all joints.
    pose3d = pose3d - pelvis
    pose3d = (pose3d - mean)/std
    return pose3d



# def unnormalize_v1(pose3d_norm, mean, std, num_joints):
#     pelvis = pose3d_norm[:, 0, :]
#     pelvis = np.expand_dims(pelvis, axis=1)
#     pelvis = np.repeat(pelvis, num_joints, axis=1)
#     pose3d = (pose3d_norm - pelvis) * std + mean
#     return pose3d


def unnormalize(pose3d_norm, mean, std, num_joints):
    pose3d = pose3d_norm * std + mean
    pelvis = pose3d[:, 0, :]
    pelvis = np.expand_dims(pelvis, axis=1)
    pelvis = np.repeat(pelvis, num_joints, axis=1)
    pose3d = pose3d + pelvis
    return pose3d

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def align_by_pelvis(joints):
    pelvis = joints[0, :]
    return joints - np.tile(np.expand_dims(pelvis, axis=0),[17,1])

def get_error(pose3d_pred, pose3d_gt):
    errors, errors_pa = [], []
    # iterate over every person
    for i in range(pose3d_pred.shape[0]):
        gt3d = pose3d_gt[i,:].reshape(-1,3)
        pred = pose3d_pred[i,:].reshape(-1,3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))
    
    errors = np.array(errors).mean()
    errors_pa = np.array(errors_pa).mean()

    return errors, errors_pa


def generate_submission(predictions, out_path):
    '''
    Generate result file for submission.
    :param predictions: predicted 3D joints in shape [N_Batch, 51]
    :param out_path: path to store the result file
    '''
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

    predictions = np.hstack([ids, predictions])

    joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")
        header.append(j + "_z")

    header = ",".join(header)
    np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')

class Argmax(nn.Module):
    """
	Module to extract coordinates from heatmaps.
	Can switch between soft (differentiable) and hard (non-differentiable)
	"""

    def __init__(self, SOFTARGMAX):
        super(Argmax, self).__init__()
        self.get_coordinates = self._softargmax if SOFTARGMAX else self._hardargmax

    def _hardargmax(self, maps: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """
		Converts 2D Heatmaps to coordinates.
		(NOTE: Recheck the mapping function and rescaling heuristic.)
		Arguments:
		    maps (torch.Tensor): 2D Heatmaps of shape (BATCH_SIZE, num_joins, dim, dim)
                    dim (int): Spatial dimension of map. Default = 64.
		Returns:
		    z (torch.Tensor): Coordinates of shape (BATCH_SIZE, num_joints*2)
		"""
        _, idx = torch.max(maps.flatten(2), 2)
        # x, y = idx / 16 + 2, torch.remainder(
        #     idx, 64) * 4 + 2  # Rescaling to (256, 256)
        x, y = (idx // 64) * 4 + 2.5, (idx % 64) * 4 + 2.5
        z = torch.stack((x, y), 2).flatten(1).float()
        return z

    def _softargmax(self, maps: torch.Tensor, beta: float = 1e7,
                    dim: int = 64) -> torch.Tensor:
        """
		Applies softargmax to heatmaps and returns 2D (x,y) coordinates
		Arguments:
			maps (torch.Tensor): 2D Heatmaps of shape (BATCH_SIZE, num_joint, dim, dim)
			beta (float): Exponentiating constant. Default = 100000
			dim (int): Spatial dimension of map. Default = 64
		Returns:
			# values (torch.Tensor): max value of heatmap; shape (BATCH_SIZE, num_joints)
			regress_coord (torch.Tensor): (x, y) co-ordinates of shape (BATCH_SIZE, num_joints*2)
		"""
        batch_size, num_joints = maps.shape[0], maps.shape[1]
        flat_map = maps.view(batch_size, num_joints, -1).float()
        Softmax = nn.Softmax(dim=-1)
        softmax = Softmax(flat_map * beta).float()
        # values = torch.sum(flat_map * softmax, -1)
        posn = torch.arange(0, dim * dim).repeat(batch_size, num_joints, 1)
        idxs = torch.sum(softmax * posn.float().to(DEVICE), -1)
        # x, y = (idxs / dim) * 4 + 2, torch.remainder(idxs, dim) * 4 + 2
        y = idxs % 64
        x = (idxs - y) / 64
        y = y * 4 + 1.5
        x = x * 4 + 1.5
        # x, y = (idxs // 64) * 4 + 2.5, (idxs % 64) * 4 + 2.5
        regress_coord = torch.stack((x, y), 2).float()
        regress_coord = regress_coord.flatten(1)
        return regress_coord

    def forward(self, x):
        return self.get_coordinates(x)
