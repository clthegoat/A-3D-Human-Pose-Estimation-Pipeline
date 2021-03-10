# A 3D Human Pose Estimation Pipeline
This repo contains the code for our Machine Perception project. <br> Developed by [Le Chen](https://github.com/clthegoat) and [Zimeng Jiang](https://github.com/zimengjiang).

You can find the report [**here**](Report.pdf).

### 1. Install dependencies
Please install all dependencies.
```
pip install numpy
pip install matplotlib
pip install scikit-image
pip install Pillow
pip install opencv-python
pip install torch torchvision
pip install tensorflow==1.15
pip install tensorflow-gpu==1.15
pip install h5py
pip install tqdm
pip install patool
pip install scipy
pip install functools
pip install pycopy-xml.etree.ElementTree
```
(Sorry if there is something missing.) 

### 2. Set up the data root
You need to follow directory structure of the `data_path` as below. Please modify **configuration.py** and put the TFRecord files in the corresponding directories!
```
${ROOT:data_path}
├── source_data
├── ├── h36
|   `── ├── TFRecord files
├── ├── mpii
|   `── ├── TFRecord files
├── extracted_data
├── ├── h36
|   `── ├── images
|       ├── images_augmented
|       ├── intrinsics
|       ├── intrinsics_univ
|       ├── pose2d
|       ├── pose3d
├── ├── mpii
|   `── ├── images
|       ├── pose2d
|       ├── visibility
├── test_data
├── ├── images
├── ├── test_tfr
|   `── ├── TFRecord files
```

For data augmentation, please download the Pascal VOC dataset by:
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
```
Then set the VOC_DATASET in **configuration.py**.

### 3. Generate data
After setting up the data root, please run:
```
python data_generator.py
```
Then you can find the extracted data in the corresponding directories.

### 4. Data augmentation
When you find the extracted images in the `images` folder, please run:
```
python data_augmentor.py
```
Then you can find the images with occluders in the `images_augmented` folder.

### 5. Training
You can train each module separately. No need to train in order.<br> 
*  From image to 2D keypoints (HRNet) 
```
python train_2d_hrnet.py
```
*  From 2D to 3D (Approach 1: Feed-forward)
```
python train_3d_feed_forward.py
```
*  From 2D to 3D (Approach 2: SemGCN)
```
python train_3d_semgcn.py
```

### 6. Evaluation
You can find the saved checkpoint in `model_parameter`. Then run:
```
python submission_generator.py
```
You can indicate the specific checkpoint you want to evaluate by modifying **submission_generator.py**.

### 7. Code references:
Part of the code is based on those repositories.
* [Sample Code](https://ait.ethz.ch/teaching/courses/2020-SS-Machine-Perception/downloads/projects/project2_skeleton.zip)
* [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git)
* [SemGCN](https://github.com/garyzhao/SemGCN.git)
* [Synthetic Occlusion](https://github.com/isarandi/synthetic-occlusion.git)
* [PoseNet](https://github.com/meetvora/PoseNet.git)
