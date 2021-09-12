# clifter slam
clifter slam is a fully differentiable dense SLAM framework. It provides a repository of differentiable building blocks for a dense SLAM system, such as differentiable nonlienar least squares solvers, differentiable ICP (iterative closest point) techniques, differentiable raycastring modules, and differentiable mapping / fusion blocks. One can use these blocks to construct SLAM systems that allow gradients to flow all the way from the outputs of the system (map, trajectory) to the inputs (raw color / depth images, parameters, calibration)

[repository](https://github.com/slowy07/clifter_slam)

## demo testing
- [pointcloud_clifter_slam](https://colab.research.google.com/drive/1QQQQ7XDop8JLL7uMeZ3FSCy6OhNk7CW0)

## documentation
[documentation](documentation/install.md)

## tutorials
- [getting started with pointclouds](documentation/Pointclouds.md)
- [getting started with RGBDImages](documentation/RGBDImages.md)


## API reference
- [``clifter_slam.config``](api_reference/clifter_slam_config.md)
- [``clifter_slam.datasets``](api_reference/clifter_slam_datasets.md)
- [``clifter_slam.geometry``](api_reference/clifter_slam_geometry.md)