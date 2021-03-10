import os
import torch

import models
import config
import util
import models.hrnet
import config.pose_hrnet
import config.hr_gcn

import numpy as np
from configuration import *
from models.hrnet import *
from models_gcn.sem_gcn import SemGCN
from models import feed_forward
from dataset import h36_test


def test_hrnet_fforward(hrnet, fforward, argmax, test_loader, p3d_mean, p3d_std):
    hrnet.eval()
    fforward.eval()
    for idx, img in enumerate(test_loader):

        # forward
        img = img.to(DEVICE)
        heatmap = hrnet(img)
        pred_p2d = argmax(heatmap)
        pred = fforward(pred_p2d)
        pred_p3d = pred.cpu().detach().numpy().reshape([BATCHSIZE_TEST, -1, 3])
        pred_p3d_unnorm = util.unnormalize(pose3d_norm=pred_p3d,
                                           mean=p3d_mean,
                                           std=p3d_std,
                                           num_joints=17)
        pred_p3d_unnorm = pred_p3d_unnorm.reshape([-1, 51])
        # record
        if not idx:
            predictions = pred_p3d_unnorm
        else:
            predictions = np.concatenate([predictions, pred_p3d_unnorm], axis=0)

    return predictions


def test_hrnet_semgcn(hrnet, semgcn, argmax, test_loader, p3d_mean, p3d_std):
    hrnet.eval()
    semgcn.eval()
    for idx, img in enumerate(test_loader):
        img = img.to(DEVICE)

        # forward
        pred_heatmap = hrnet(img)
        pred_p2d = argmax(pred_heatmap).view([-1, 17, 2])
        pred_p3d = semgcn(pred_p2d).view([BATCHSIZE_TEST, -1]).cpu().detach().numpy().reshape([BATCHSIZE_TEST, -1, 3])
        pred_p3d_unnorm = util.unnormalize(pose3d_norm=pred_p3d,
                                           mean=p3d_mean,
                                           std=p3d_std,
                                           num_joints=17)
        pred_p3d_unnorm = pred_p3d_unnorm.reshape([-1, 51])

        # record
        if not idx:
            predictions = pred_p3d_unnorm
        else:
            predictions = np.concatenate([predictions, pred_p3d_unnorm], axis=0)

    return predictions


def main():

    TYPE = 'hrnet_feedforward'

    # read data
    img_list = sorted(os.listdir(FILE_H36_IMG_TEST))
    test_set = h36_test(list=img_list,
                        transfrom=TRANSFORM)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCHSIZE_TEST,
        shuffle=False,
        num_workers=8,
    )

    # for normalization
    pose3d_mean = np.load('./misc/mean.npy')
    pose3d_std = np.load('./misc/std.npy')
    pose3d_std[0, :] = 1

    if TYPE == 'hrnet_feedforward':

        # load model
        hrnet = PoseHighResolutionNet(config.hr_gcn).to(DEVICE).float()
        hrnet.load_state_dict(torch.load(MODEL_PATH + 'hrnet.pkl'))
        ff = feed_forward.FeedForward().to(DEVICE)
        ff.load_state_dict(torch.load(MODEL_PATH + 'feedforward.pkl'))

        # softmax
        argmax = util.Argmax(config.hr_gcn.SOFTARGMAX).to(DEVICE)

        # prediction
        prediction = test_hrnet_fforward(hrnet=hrnet,
                                        fforward=ff,
                                        argmax=argmax,
                                        test_loader=test_loader,
                                        p3d_mean=pose3d_mean,
                                        p3d_std=pose3d_std)

        util.generate_submission(prediction, SUBMISSION)

    else:
        # load model
        hrnet = PoseHighResolutionNet(config.hr_gcn).to(DEVICE).float()
        hrnet.load_state_dict(torch.load(MODEL_PATH + '2hrnet.pkl'))
        semgcn = SemGCN(adj=config.hr_gcn.ADJ,
                    hid_dim=config.hr_gcn.HID_DIM,
                    num_layers=config.hr_gcn.NUM_LAYERS,
                    p_dropout=None,
                    nodes_group=None).to(DEVICE).float()
        # semgcn.load_state_dict(torch.load(MODEL_PATH + SEMGCN))
        semgcn.load_state_dict(torch.load(MODEL_PATH + '2semgcn.pkl'))

        # softmax
        argmax = util.Argmax(config.hr_gcn.SOFTARGMAX).to(DEVICE)

        # prediction
        prediction = test_hrnet_semgcn(hrnet=hrnet,
                                       semgcn=semgcn,
                                       argmax=argmax,
                                       test_loader=test_loader,
                                       p3d_mean=pose3d_mean,
                                       p3d_std=pose3d_std)
        util.generate_submission(prediction, SUBMISSION)
    print("Submission generated!")


if __name__ == '__main__':
    main()