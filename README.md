# clifter_slam
clifter slam is a fully differentiable dense SLAM framework. It provides a repository of differentiable building blocks for a dense SLAM system, such as differentiable nonlienar least squares solvers, differentiable ICP (iterative closest point) techniques, differentiable raycastring modules, and differentiable mapping / fusion blocks. One can use these blocks to construct SLAM systems that allow gradients to flow all the way from the outputs of the system (map, trajectory) to the inputs (raw color / depth images, parameters, calibration)

## documentation
[documentation](documentation/install.md)

## tutorials
- [getting started with pointclouds](documentation/Pointclouds.md)
- [getting started with RGBDImages](documentation/RGBDImages.md)