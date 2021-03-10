import os
import torch
import config.pose_hrnet
import util
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dataset import h36
from configuration import *
from models import feed_forward

def validation(model, val_loader,p3d_mean, p3d_std):
    joint_error = []
    PA_MPJPE = []
    with torch.no_grad():
        model.eval()
        for _, (_, pose3d, _, pose2d, _) in enumerate(val_loader):
            p2d = pose2d.to(DEVICE).float()

            p3d_pred = model(p2d)

            # unnormalize groundtruth pose3d, after unnorm, pelvis = [0,0,0]
            p3d_gt_unnorm = pose3d.cpu().detach().numpy().reshape([BATCHSIZE, -1, 3])
            p3d_pelvis = p3d_gt_unnorm[:, 0, :]
            p3d_pelvis = np.expand_dims(p3d_pelvis, axis=1)
            p3d_pelvis = np.repeat(p3d_pelvis, 17, axis=1)
            p3d_gt_unnorm = p3d_gt_unnorm - p3d_pelvis

            # unnormalize predicted pose3d
            p3d_pred_np = p3d_pred.cpu().detach().numpy().reshape([BATCHSIZE, -1, 3])
            p3d_pred_unnorm = util.unnormalize(pose3d_norm=p3d_pred_np,
                                                mean=p3d_mean,
                                                std=p3d_std,
                                                num_joints=17)

            MPJPE = util.get_error(pose3d_pred=p3d_pred_unnorm, pose3d_gt=p3d_gt_unnorm)
            joint_error.append(MPJPE[0])
            PA_MPJPE.append(MPJPE[1])
    joint_error_mean = np.array(joint_error).mean()
    PA_MPJPE_mean = np.array(PA_MPJPE).mean()
    return joint_error_mean, PA_MPJPE_mean


def train(model, train_loader, val_loader, optimizer, p3d_mean, p3d_std, traincounter):
    model.train()
    val_joint_error_mean=1000
    val_PA_MPJPE_mean=1000
    loader = tqdm(train_loader)

    for _, (_, pose3d, pose3d_norm, pose2d, _) in enumerate(loader):
        # learn normalized pose3d
        p3d_gt_normed, p2d = pose3d_norm.to(DEVICE), pose2d.to(DEVICE).float()

        # forward
        p3d_pred = model(p2d)
        # for now only use 3d mse loss
        loss = F.mse_loss(p3d_gt_normed, p3d_pred, reduction='mean')

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # unnormalize groundtruth pose3d, after unnorm, pelvis = [0,0,0]
        p3d_gt_unnorm = pose3d.cpu().detach().numpy().reshape([BATCHSIZE, -1, 3])
        p3d_pelvis = p3d_gt_unnorm[:, 0, :]
        p3d_pelvis = np.expand_dims(p3d_pelvis, axis=1)
        p3d_pelvis = np.repeat(p3d_pelvis, 17, axis=1)
        p3d_gt_unnorm = p3d_gt_unnorm - p3d_pelvis

        # unnormalize predicted pose3d
        p3d_pred_np = p3d_pred.cpu().detach().numpy().reshape([BATCHSIZE, -1, 3])
        p3d_pred_unnorm = util.unnormalize(pose3d_norm=p3d_pred_np,
                                              mean=p3d_mean,
                                              std=p3d_std,
                                              num_joints=17)

        traincounter += 1
        if not traincounter % 1000:
            torch.save(model.state_dict(), MODEL_PATH + FEEDFORWARD)
        if not traincounter % 5000:
            val_joint_error_mean, val_PA_MPJPE_mean = validation(model=model, val_loader=val_loader, p3d_mean=p3d_mean, p3d_std=p3d_std)
            print('val_joint_error_mean: {}, val_PA_MPJPE_mean: {}'.format(val_joint_error_mean, val_PA_MPJPE_mean))
        MPJPE = util.get_error(pose3d_pred=p3d_pred_unnorm, pose3d_gt=p3d_gt_unnorm)
        loader.set_description("joint error:{:.4f}, MPJPE: {:.4f}".format(MPJPE[0],MPJPE[1]))
        loader.refresh()

    return traincounter


def main():
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    use_all_subject = True

    # Basic setting
    img_list = sorted(os.listdir(FILE_H36_IMG_aug))
    if use_all_subject: 
        # use all subject to train 
        train_list = [img for img in img_list]
        train_list = [idx.split('.jpg')[0] for idx in train_list]
        train_len = len(train_list)
        train_length = int(0.9 * train_len)
        val_length = train_len - train_length
    else:
        train_subject_id = np.array([1, 5, 6, 7])
        val_subject_id = np.array([8])
        # train list setting
        train_list = [img for img in img_list if np.any(train_subject_id == int(img.split('_')[2]))]
        train_list = [idx.split('.jpg')[0] for idx in train_list]
        train_length = len(train_list)
        # val list setting
        val_list = [img for img in img_list if np.any(val_subject_id == int(img.split('_')[2]))]
        val_list = [idx.split('.jpg')[0] for idx in val_list]
        val_length = len(val_list)

    # mean and std
    pose3d_mean = np.load('./misc/mean.npy')
    pose3d_std = np.load('./misc/std.npy')
    pose3d_std[0, :] = 1

    # Load trainset
    if use_all_subject:
        data_set = h36(list=train_list,
                        length=train_len,
                        transform=TRANSFORM,
                        pose3d_mean=pose3d_mean,
                        pose3d_std=pose3d_std)

        train_set, val_set = torch.utils.data.random_split(data_set,
                                                        [train_length, val_length])

    else:
        train_set = h36(list=train_list,
                    length=train_length,
                    transform=TRANSFORM,
                    pose3d_mean=pose3d_mean,
                    pose3d_std=pose3d_std)
        
        val_set = h36(list=val_list,
                    length=val_length,
                    transform=TRANSFORM,
                    pose3d_mean=pose3d_mean,
                    pose3d_std=pose3d_std)


    train_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=8,
                                            drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=8,
                                            drop_last=True)

    model = feed_forward.FeedForward().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=5e-4,
                                 weight_decay=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                3,
                                gamma=0.85,
                                last_epoch=-1)  # optional
    epoch = 0
    traincounter = 0

    # train
    while epoch < 20:
        # scheduler.step()
        epoch += 1
        print('start epoch:', epoch)
        traincounter = train(model=model,
                             train_loader=train_loader,
                             val_loader = val_loader,
                             optimizer=optimizer,
                             p3d_mean=pose3d_mean,
                             p3d_std=pose3d_std,
                             traincounter=traincounter
                             )
        torch.save(model.state_dict(), MODEL_PATH + str(epoch) + FEEDFORWARD)

if __name__ == '__main__':
    main()