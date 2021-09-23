# clifter_slam.slam
## clifter_slam.slam.icpslam

```python
class ICPSLAM(
    *,
    odom: str = `gradicp`,
    dsratio: int = 4,
    numiters: int = 20,
    damp: float = 1e-08,
    dist_thresh: Optional[Union[float, int]] = 1.0,
    nu: Union[float, int] = 200.0,
    device: Optional[Union[torch.Device, str]] = None
)
```
icp slam for batched sequences of RGB-D images

- parameters
    - odom(``str``)

        odometry method to be used from {``gt``, ``icp``, ``gradicp``}. default ``gradicp``

    - dsratio(``int``)

        downsampling ratio to apply to input frames before ICP.Only used if odom is ``icp`` or ``gradicp`` default ``4``

    - numiters(``int``)

        number of iterations to run the optimization for. only used if __odom__ is ``icp`` or ``gradicp``, default 20

    - damp(``float`` or ``torch.Tensor``)

        damping coefficient for nonlinear least-sequares. only used if __odom__  is ``icp`` or ``gradicp`` default ``1e-8``

    - dist_thresh(``float`` or ``int`` or ``None``)

        distance threshold for removing __src_pc__ points distant from __tgt_pc__ . only used if __odom__ is ``icp`` or ``gradicp`` default ``None``

    - lambda_max(``float`` or ``int``)

        maximum value the damping function can assume (``lambda_min`` will be ``1 / lambda_max``) only used if __odom__ is ``gradicp``

    - B(``float`` or ``int``)
    
        clifterLM asfallof control paramter

    - B2(``float`` or ``int``)

        clifterLM control parameter. only used if __odom__ is ``icp``


example
```
>>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
>>> slam = ICPSLAM(odom='gt')
>>> pointclouds, poses = slam(rgbdimages)
>>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])
```

```python
forward(
    frames: clifter_slam.structures.rgdbimages.RGBDImages
)
```
builds global map pointclouds from a batch of input RGBDImages with a batch size of __B__ and sequence length of __L__

- paramter
    - frames (``clifter_slam.RGBDImages``)

        input batch of frames with a sequence length of __L__

    - return

        tuple containing:

        - pointclouds(``clifter_slam.Pointclouds``) pointclouds object containing ``B`` global maps

        - poses(``torch.Tensor``) poses computed by the odometry method

    - return type

        ``tuple``

```python
step(
    pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
    live_frame: Optional[clifter_slam.structures.rgbdimages.RGBDImages] = None,
    inplace: bool = False
)
```
updates global maps pointclouds with a SLAM step on __live_frame__. of __prev_frame__ is not None, computes the relative transformation between __live_frame__ and __prev_frame__ using the selected odometry provider.if __prev_frame__ is None, the poses from __live_frame__.

- parameter
    - pointclouds(``clifter_slam.Pointclouds``)
        input batch of pointcloud global map

    - live_frame(``clifter_slam.RGBDImages``)

        input batcj of live frames(at time step __t__). must have sequence length of 1

    - prev_frame(``clifter_slam.RGBDImages`` or ``None``)

        input batch of previous frames (at step __t__ - 1) must have sequence length of 1. if None will skip calling odometry provider and use the pose __live_frame__. default None

    - inplace(``bool``)

        can optionally update the pointclouds and __live_frame__ poses in place. default ``False``

    - return

        tuple containing
            - pointclouds(``clifter_slam.Pointclouds``) update global maps
            - poses(``torch.Tensor``) poses for the love_frame batch

    - return type

        ``tuple``
    

## clifter_slam.slam.poinfusion

```python
class POintFusion(
    *,
    odom: str = 'gradicp',
    dist_th = Union[float, int] = 0.05,
    angle_th = Union[float, int] = 20,
    sigma = Union[float, int] = 0.6,
    dsratio: int = 4,
    numiters: int = 20,
    damp: float = 1e-08,
    dist_thresh: Optiona;[Union[float, int]] = 1.0,
    B2: Union[float, int] = 1.0,
    nu: Union[float, int] = 200.0,
    device: Optiona;[Union[torch.device, str]] = None
)
```

point-based Fusion (pointfusion for short) SLAM for batched sequences of RGB-D images

- parameters
    - odom(``str``)

        odometry method to be used form {``gt``, ``icp``, ``gradicp``}.defaukt ``gradicp``

    - dist_th(``float`` or ``int``)

        distance of threshold
    
    - dot_th(``float`` or ``int``)

        dot prodcut threshold

    - sigma(``torch.Tensor`` or ``float`` or ``int``)

        width of the gaussian bell. Original paper uses 0.6 emperically

    - dscratio(``int``)

        downsampling ratio to apply to input frames before ICP. only used if __odom__ is ``icp``.
        default ``4``

    - numiters(``int``)

        number of iterations to run the optimization fo. only used __odom__ is ``icp``. default 20

    - damp(``float`` or ``torch.Tensor``)

        damping coefficient for nonllinear lesat-squares. only used if odom is ``icp``. default 1e-8

    - dist_thresh(``float`` or ``int`` or ``None``)

        distance threshold for removing __src_pc__ point distant from __tgt_pc__ only used if __odom__ is ``icp``. default ``None``
        
    - lambda_ma(``float`` or ``int``)

        maximum value the damping function can assume (``lambda_min`` will be ``1 / lambda_max``)

    - B(``float`` or ``int``)

        clifterLM fallof control parameter

    - B2(``float`` or ``int``)

        clifterLM control parameter

    - nu(``float`` or ``int``)

        clifterLM acontrol parameter

    - device(``torch.device`` or ``str`` or ``None``)

        the desired device of internal tensor. if None, sets device to abe the CPU, default None

example
```
>>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
>>> slam = PointFusion(odom='gt')
>>> pointclouds, poses = slam(rgbdimages)
>>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])
```


## clifter_slam.slam.fusionutils

```python
update_map_fusion(
    pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
    rgbdimages: clifter_slam.structures.rgbdimages.RGBDImages,
    dist_th: Union[floatm int],
    sigma: Union[torch.Tensor, float, int],
    inplace: bool = False
) -> clifter_slam.structures.poinclouds.Pointclouds
```

update pointclouds in place given the live frame RGB-D image using PointFusion

- parameters

    - pointclouds(``clifter_slam.pointclouds``)

        pointclouds of global maps must have points, color, normals and features
    
    - rgbdimages(``clifter_slam.RGBDImages``)

        live frames from the latest sequnce

    -  dist_th(``float`` or ``int``)

        distance threshold

    - dot_th(``float`` or ``int``)

        dot prodcut threshold

    - sigma(``torch.Tensor`` or ``float`` or ``int``)

        standard deviation of the gaussian

    - inplace(``bool``)

        can optionally update the pointclouds in-place, default: False

- returns

    update Pointclouds object containing global maps

- return type

    ``clifter_slam.Pointclouds``


```python
update_map_aggreate(
    pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
    rgbdimages: clifter_slam.structures.rgbdimages.RGBDImages,
    inplace: bool = False,
) -> clifter_slam.structures.pointclouds.Pointclouds
```

aggregate points from live frames with global maps by appending the live frame points

- parameters

    - pointclouds(``clifter_slam.Pointclouds``)

        pointclouds of the aglobal maps. must have points,colors, normals and features

    - rgbdimages(``cliterslam.RGBDImages``)

        live frames fromthe latest sequence

    - inplace(``bool``)

        can optionally update the pointclouds in-place. default False

- return

    update pointclouds object containing globla maps

- return type

    ``clifter_slam.Pointclouds``