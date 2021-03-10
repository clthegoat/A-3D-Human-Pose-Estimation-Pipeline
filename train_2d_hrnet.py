import os
import torch
import models
import config
import models.hrnet
import config.pose_hrnet
from tqdm import tqdm
from dataset import h36
from configuration import *
from loss import *


def validation(model, val_loader, lossfunc):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for idx, (img, pose3d, pose3d_norm, pose2d, heatmap) in enumerate(val_loader):
            img, heatmap = img.to(DEVICE), heatmap.to(DEVICE)
            pred = model(img)
            loss = lossfunc(pred, heatmap)
            val_loss += loss.item()
    val_loss_mean = val_loss / len(val_loader)
    return val_loss_mean


def train(model, train_loader, val_loader, optimizer, lossfunc, traincounter):
    loader = tqdm(train_loader)
    train_loss = 0
    val_loss_mean = 1000
    for idx, (img, pose3d, pose3d_norm, pose2d, heatmap) in enumerate(loader):
        model.train()
        img, heatmap = img.to(DEVICE), heatmap.to(DEVICE)

        # Forward
        pred = model(img)
        loss = lossfunc(pred, heatmap)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # For recording
        traincounter += 1
        if not traincounter % 1000:
            torch.save(model.state_dict(), MODEL_PATH + HRNET)
        if not traincounter % 5000:
            val_loss_mean = validation(model=model,
                                   val_loader=val_loader,
                                   lossfunc=lossfunc)
            print('val_loss_mean: {}'.format(val_loss_mean))
        loader.set_description("train loss: {:.8f}".format(train_loss / (idx + 1)))
        loader.refresh()
    return traincounter


def get_new_HR2D() -> nn.Module:
    """
	Returns an instance of HRNet with new final Conv2d layer
	"""
    model = models.hrnet.PoseHighResolutionNet(config.pose_hrnet)
    model.init_weights(config.hrnet.BASE_WEIGHTS, config.hrnet.USE_GPU)
    for param in model.parameters():
        param.requires_grad = True
    return model
    

def main():
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

        train_list = [img for img in img_list if np.any(train_subject_id == int(img.split('_')[2]))]
        train_list = [idx.split('.jpg')[0] for idx in train_list]
        train_length = len(train_list)

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
                                            batch_size=128,
                                            shuffle=True,
                                            num_workers=8,
                                            drop_last=True)
    model = get_new_HR2D().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-5,
                                 weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                step_size=1,
                                gamma=0.1,
                                last_epoch=-1)  # optional
    lossfunc = JointsMSELoss()

    # For recording
    epoch = 0
    traincounter = 0
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    while epoch < 3:
        # lr_scheduler.step()
        epoch += 1
        print('start training epoch:', epoch)
        traincounter = train(model=model,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             optimizer=optimizer,
                             lossfunc=lossfunc,
                             traincounter=traincounter)
        scheduler.step()
        torch.save(model.state_dict(), MODEL_PATH + str(epoch)+HRNET)


if __name__ == '__main__':
    main()
