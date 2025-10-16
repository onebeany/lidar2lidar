

## Build

```
cd ~/your_workspace/src
git clone https://github.com/onebeany/lidar2lidar.git
cd ~/your_workspace
catkin build
```

## Launch

$ `roslaunch lidar2lidar visualize_lidar2lidar.launch use_refactored:=true`

Refactoring is not finished yet, but you can test scan registration with this version. If you want to use original version, change `use_refactored` to `false`

## Description

### For scan registraion part

1. Refactored version: `performImprovedAlignment` in `visualize_lidar2lidar_refactored.cpp`.
2. Original version from Kevin: `lidar2_callback` in `visualize_lidar2lidar.cpp`.

### About the topic 

#### Refactored version

1. `/lidar1_transformed`: transformed point cloud of front LiDAR (LiDAR 1) by extrinsic
2. `/lidar1_cropped_ROI`: based on the range of `/lidar2_kinematics`, `/lidar1_transformed` is cropped
3. `/lidar2_kinematics`: transformed point cloud of upper LiDAR (LiDAR 2) by kinematics
4. `/lidar2_coarse_ICP`: transformed point cloud of front LiDAR by result transformation matrix from coarse **ICP**
5. `/lidar2_fine_ICP`: transformed point cloud of front LiDAR by result transformation matrix from fine **ICP**
6. `/lidar2_gicp`: transformed point cloud of front LiDAR by result transformation matrix from from **GICP**
7. `/lidar2_transformed`: Final result of LiDAR 2 point cloud. It depends on success or not of each process (ICP, GICP). 
    - If coarse ICP failed, it will be `/lidar2_kinematics`.
    - ...
    - If GICP failed, it will be `/lidar2_fine_ICP`
8. `/cloud_merged`: `/lidar1_transformed` + `/lidar2_transfromed`.

#### Original version

1. `/lidar1_transformed`: transformed point cloud of front LiDAR (LiDAR 1) by extrinsic
2. `/lidar2_transformed`: `/lidar1_transformed` + LiDAR2 point cloud which is finally transformed by the result transformation from ICP or GICP...

## ETC

`visualize_lidar2lidar.cpp` includes whole code itself, but `visualize_lidar2lidar_refactored.cpp` doesn't. Functions and classes are being separated...

## TODO

- implement this with only GICP
- Change the standard of judgement of well success of ICP/GICP, from whether converged or not to fitness score
