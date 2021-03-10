import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from configuration import *


def _parse_h36(example_proto):

    image_feature_description = {
        # *image : images in size 256x256, cropped from the original images and scaled. Human bodies are located at the image center.
        'image': tf.io.FixedLenFeature([], tf.string),
        # offset : parameter for cropping. Each pixel in image corresponds to [pixel / scale + offset in the original image].
        'offset': tf.io.FixedLenFeature([], tf.string),
        # scale : parameter for cropping. Each pixel in image corresponds to [pixel / scale + offset in the original image].
        'scale': tf.io.FixedLenFeature([], tf.float32),
        # *pose2d_crop : 2D pixel coordinates of 17 joints in cropped images. [pose2d_crop = (pose2d - offset) * scale]
        'pose2d_crop': tf.io.FixedLenFeature([], tf.string),
        # pose2d : 2D pixel coordinates of 17 joints in original images. [pose2d = intrinsics * pose3d]
        'pose2d': tf.io.FixedLenFeature([], tf.string),
        # *pose3d :  3D positions of 17 joints in original camera frames. [pose2d = intrinsics * pose3d]
        'pose3d': tf.io.FixedLenFeature([], tf.string),
        # pose3d_univ :  3D positions of 17 joints in a canonical world coordiante frame (from MoCap). [pose2d = intrinsics_univ * pose3d_univ]
        'pose3d_univ': tf.io.FixedLenFeature([], tf.string),
        # intrinsics : matrix of the camera for projecting pose3d to pose2d. [pose2d = intrinsics * pose3d]
        'intrinsics': tf.io.FixedLenFeature([], tf.string),
        # intrinsics_univ : matrix for projecting pose3d_univ to pose2d.. [pose2d = intrinsics_univ * pose3d_univ]
        'intrinsics_univ': tf.io.FixedLenFeature([], tf.string),
        # framd_id : frame ID.
        'framd_id': tf.io.FixedLenFeature([], tf.int64),
        # camera : The data is capture using 4 cameras. This is the camera ID.
        'camera': tf.io.FixedLenFeature([], tf.int64),
        # *subject : ID of the person in the image. [1,5,6,7,8]
        'subject': tf.io.FixedLenFeature([], tf.int64),
        # action : ID of the action performed in the image.  2-Directions,3-Discussion, 4-Eating, 5-Greeting, 6-Phoning, 7-Posing, 8-Purchases, 9-Sitting, 10-Sitting Down, 11-Smoking, 12-Taking Photo, 13-Waiting, 14-Walking, 15-Walking Dog, 16-Walk Together
        'action': tf.io.FixedLenFeature([], tf.int64),
        # subaction : Each action contains several sequences. This is the sequence ID.
        'subaction': tf.io.FixedLenFeature([], tf.int64),
    }

    sample = tf.io.parse_single_example(example_proto,
                                        image_feature_description)

    sample['pose3d'] = tf.io.decode_raw(sample['pose3d'], tf.float32)
    sample['pose3d'] = tf.reshape(sample['pose3d'], (17, 3))

    sample['pose2d_crop'] = tf.io.decode_raw(sample['pose2d_crop'], tf.float32)
    sample['pose2d_crop'] = tf.reshape(sample['pose2d_crop'], (17, 2))

    sample['intrinsics'] = tf.io.decode_raw(sample['intrinsics'], tf.float32)
    sample['intrinsics'] = tf.reshape(sample['intrinsics'], (1, 4))

    sample['intrinsics_univ'] = tf.io.decode_raw(sample['intrinsics_univ'], tf.float32)
    sample['intrinsics_univ'] = tf.reshape(sample['intrinsics_univ'], (1, 4))

    sample['image'] = tf.io.decode_raw(sample['image'], tf.uint8)
    sample['image'] = tf.reshape(sample['image'], (256, 256, 3))

    return sample['image'], sample['pose3d'], sample['pose2d_crop'], sample['intrinsics'], sample['intrinsics_univ'], \
           sample['framd_id'], sample['camera'], sample['subject'], sample['action'], sample['subaction']


def _parse_mp2(example_proto):

    feature_map = {
        # encoded : encoded images which can be decoded to raw image. See code below.
        'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        # pose2d : 2D pixel coordinates of joints.
        'pose2d': tf.FixedLenFeature([], dtype=tf.string),
        # visibility : 2D joint visibility. 0 means the joint is not visible hence the joint position annotation is
        # invalid.
        'visibility': tf.FixedLenFeature((1, 16), dtype=tf.int64),
    }
    sample = tf.io.parse_single_example(example_proto,
                                        feature_map)

    sample['image'] = tf.image.decode_jpeg(sample['image'], channels=3)
    sample['image'] = tf.cast(sample['image'], tf.uint8)

    sample['pose2d'] = tf.io.decode_raw(sample['pose2d'], tf.float32)
    sample['pose2d'] = tf.reshape(sample['pose2d'], (16, 2))

    sample['vis'] = tf.cast(sample['visibility'], dtype=tf.float32)

    return sample['image'], sample['pose2d'], sample['vis']


def _parse_test(example_proto):

    image_feature_description = {
        'image':
            tf.io.FixedLenFeature([], tf.string),
        'offset':
            tf.io.FixedLenFeature([], tf.string),
        'scale':
            tf.io.FixedLenFeature([], tf.float32),
        'intrinsics':
            tf.io.FixedLenFeature([], tf.string),
        'intrinsics_univ':
            tf.io.FixedLenFeature([], tf.string),
        'camera':
            tf.io.FixedLenFeature([], tf.int64),
        'subject':
            tf.io.FixedLenFeature([], tf.int64),
        'action':
            tf.io.FixedLenFeature([], tf.int64),
        'subaction':
            tf.io.FixedLenFeature([], tf.int64),
    }

    sample = tf.io.parse_single_example(example_proto,
                                        image_feature_description)

    sample['image'] = tf.io.decode_raw(sample['image'], tf.uint8)
    sample['image'] = tf.reshape(sample['image'], (256, 256, 3))

    # return sample['image'], sample['pose3d'], sample['pose2d_crop']
    return sample['image'], sample['intrinsics'], sample['intrinsics_univ'], \
           sample['camera'], sample['subject'], sample['action'], sample['subaction']


def int2str(num, min_len):
    strings = str(num)
    while len(strings) < min_len:
        strings = '0' + strings
    return strings


def main():
    TYPE = sys.argv[1]
    # TYPE = 'h36'
    # TYPE = 'mp2'
    # TYPE = 'test'

    if TYPE == 'h36':
        # For h36
        print('now generating h36 dataset...')
        if not os.path.exists(FILE_H36_IMG_ori):
            os.makedirs(FILE_H36_IMG_ori)
            os.makedirs(FILE_H36_POSE_3D)
            os.makedirs(FILE_H36_POSE_2D)
            os.makedirs(FILE_H36_POSE_INTR)
            os.makedirs(FILE_H36_POSE_INTR_UNI)

        
        h36_path = tf.data.Dataset.list_files(os.path.join(FILE_H36_TFR, 'h36', '*'), shuffle=True)
        # h36_path = FILE_H36_TFR
        dataset = tf.data.TFRecordDataset(filenames=h36_path,
                                          compression_type="ZLIB")
        dataset = dataset.map(_parse_h36)

        # Construct iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        image_, pose3d_, pose2d_, intrinsics_, intrinsics_univ_, frame_id_, camera_, subject_, action_, subaction_ = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            for i in range(NUM_SAMPLES_H36):
                image, pose3d, pose2d, intrinsics, intrinsics_univ, framd_id, camera, subject, action, subaction = sess.run(
                    [image_, pose3d_, pose2d_, intrinsics_, intrinsics_univ_, frame_id_, camera_, subject_, action_,
                     subaction_])
                image = Image.fromarray(image, 'RGB')

                # Save
                image.save(
                    FILE_H36_IMG_ori + str(framd_id) + '_' + str(camera) + '_' + str(subject) + '_' + str(action) + '_' + str(
                        subaction) + '.jpg')
                np.savetxt(
                    FILE_H36_POSE_3D + str(framd_id) + '_' + str(camera) + '_' + str(subject) + '_' + str(action) + '_' + str(
                        subaction) + '.txt', pose3d)
                np.savetxt(
                    FILE_H36_POSE_2D + str(framd_id) + '_' + str(camera) + '_' + str(subject) + '_' + str(action) + '_' + str(
                        subaction) + '.txt', pose2d)
                np.savetxt(
                    FILE_H36_POSE_INTR + str(framd_id) + '_' + str(camera) + '_' + str(subject) + '_' + str(action) + '_' + str(
                        subaction) + '.txt', intrinsics)
                np.savetxt(FILE_H36_POSE_INTR_UNI + str(framd_id) + '_' + str(camera) + '_' + str(subject) + '_' + str(
                    action) + '_' + str(subaction) + '.txt', intrinsics_univ)
    elif TYPE == 'mp2':
        # For mp2
        print('now generating mpii dataset...')
        if not os.path.exists(FILE_MP2_IMG):
            os.makedirs(FILE_MP2_IMG)
            os.makedirs(FILE_MP2_POSE_2D)
            os.makedirs(FILE_MP2_VIS)


        # mp2_path = FILE_MP2_TFR
        mp2_path = tf.data.Dataset.list_files(os.path.join(FILE_MP2_TFR, 'mpii', '*'), shuffle=True)
        dataset = tf.data.TFRecordDataset(filenames=mp2_path)
        dataset = dataset.map(_parse_mp2)

        # Construct iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        img, pose_2d, vis = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            for i in range(NUM_SAMPLES_MPII):

                # Read raw data
                image, pose2d, visibility = sess.run([img, pose_2d, vis])
                image = Image.fromarray(image, 'RGB')

                # Resize
                height, width = np.shape(image)[0], np.shape(image)[1]
                img_resized = image.resize((256, 256))
                pose2d[:, 0] = pose2d[:, 0] * 256. / width
                pose2d[:, 1] = pose2d[:, 1] * 256. / height

                # Save
                img_resized.save(FILE_MP2_IMG + int2str(i, 6) + '.jpg')
                np.savetxt(FILE_MP2_POSE_2D + int2str(i, 6) + '.txt', pose2d)
                np.savetxt(FILE_MP2_VIS + int2str(i, 6) + '.txt', visibility)
    elif TYPE == 'test':
        print('now generating test dataset...')
        if not os.path.exists(FILE_H36_IMG_TEST):
            os.makedirs(FILE_H36_IMG_TEST)

        # read raw data
        h36_test_path = tf.data.Dataset.list_files(os.path.join(FILE_H36_TFR_TEST, '*'), shuffle=False)

        dataset = tf.data.TFRecordDataset(filenames=h36_test_path, 
                                        compression_type="ZLIB")
        dataset = dataset.map(_parse_h36)

        # Construct iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        image_, intrinsics_, intrinsics_univ_, camera_, subject_, action_, subaction_ = iterator.get_next()

        with tf.compat.v1.Session() as sess:
            for i in range(NUM_SAMPLES_TEST):
                image, intrinsics, intrinsics_univ, camera, subject, action, subaction = sess.run(
                    [image_, intrinsics_, intrinsics_univ_, camera_, subject_, action_,
                    subaction_])
                image = Image.fromarray(image, 'RGB')

                # save images
                image.save(
                    # FILE_H36_IMG_TEST + str(camera) + '_' + str(subject) + '_' + str(action) + '_' + str(subaction) + '.jpg')
                    FILE_H36_IMG_TEST + int2str(i, 6) + '.jpg')
    else:
        raise Exception('please indicate valid file name!')


if __name__ == "__main__":
    main()
