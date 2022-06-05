# Rope3D-Toolkit
Third-party toolkit for Rope3D dataset

It is a third-party toolkit for [Rope3D Dataset](https://thudair.baai.ac.cn/rope).

In this toolkit, we will implement following functions, including:

- [x] 2d and 3d box label visualization.
- [ ] prediction results evaluation.
- [ ] training in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). 

## Data preparation
Acquire raw data of Rope3D from [here](https://thudair.baai.ac.cn/rope). Then unzip this dataset and ordered like following:
```
├── Rope3D
│   ├── training
│   │   ├── calib
│   │   ├── denorm
│   │   ├── extrinsics
|   |   ├── label_2
|   |   ├── train.txt
|   ├── training-depth_2
│   ├── training-image_2
│   ├── validation
│   │   ├── calib
│   │   ├── denorm
│   │   ├── extrinsics
|   |   ├── label_2
|   |   ├── val.txt
|   ├── validation-depth_2
│   ├── validation-image_2
```

## Dataset Visualization
```shell
ln -s /path/to/Rope3D data 
bash vis.sh
```
Visualization examples
![](./example/0_3d.jpg)

## Acknowledgement

- [rope3d-dataset-tools](https://github.com/liyingying0113/rope3d-dataset-tools)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [kitti_object_vis](https://github.com/kuixu/kitti_object_vis)
