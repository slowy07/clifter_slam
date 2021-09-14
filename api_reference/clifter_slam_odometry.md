# clifter slam odometry

## clifter_slam.odometry.base

```python
class OdometryProvider(*params)
```
base class for all odometry provides

your providers sgould also subclass this class. you shold override the provide() method

- ``abstract provide(*args, **kwargs)``

    definces the aodomotery computation performed at every __provide()__ call

## clifter_slam.odometry.gradicp

```python
celass GradICPOdometryProvider(
    numters: iknt = 20,
    damp: float = 1e-08,
    dist_thresh: Optional[Union[float, in]] = None,
    lambda_max: Union[float, int] = 2.0,
    B: Union[float, int] = 1.0,
    B2: Union[float, int] = 1.0
    nu: Union[float, int] = 200.0
)
```
an odometry provider that uses the (differentiable) gradICP technique presented in the clifter slam documentation.

```python
provide(
    maps_pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
    frames_pointclouds: clifter_slam.structures.pointclouds.PointClouds
) -> torch.Tensor
```
uses gradICP to computer the relative homogeneous transformation when applied to __frames_pointclouds__, would acause the points to align with points of __maps_pointclouds__.

- parameters
    - maps_pointclouds(``clifter_slam.Pointclouds``)

        object containing batch of map pointclouds of batch size
    
    - frames_pointclouds(``clifter_slam.Pointclouds``)

        aboject containing batch of life frame pointclouds of batch size

- returns

    the relative transformation that woulld align __maps_pointclouds__ with __frames_pointclouds__

- return type

    ``torch.Tensor``

## clifter_slam.odometry.groundtruth

```python
class GroundTruthOdometryProvider(*params)
```
ground truth odometry provider. computers the relative transformation between a pair of __clifter_slam.RGBDImages__ objects. both objects must contain poses attributes

```python
Provide(
    rgbdimages1: clifter_slam.structures.rgbdimages.RGBDImages,
    rgbdimages1: clifter_slam.structures.rgbdimages.RGBDImages
) -> torch.Tensor
```
computes the relative homogeneous transformation between poses of rgbdimages2 and rgbdimages1. the relative atansofrmation is compute T = (T1)^1 T2

- parameters
  
  - rgbdimages1(``clifter_slam.RGBDImages``)

    object containing batch of reference poses of shape (B , 1, 4, 4)

  - rgbdimages2(``clifter_slam.RGBDImages``)

    object containing batch of destination poses of shape (B, 1,4, 4)

- returns

  the relative transformation netweem the poses of rgbdimages1 and rgbdimages2 (T = (T1)^-1 T2)

- return type
  
  ``torch.Tensor``


## aclifter_slam.odometry.icp

```python
class GradICPOdometryProvider(
  numiters: int = 20,
  damp: float = 1e-08,
  dist_thresh: Optional[Union[float, int]] = None
)
```
ICP odometry provider using a point-to-plane error metric. computes the relative transformation between a pair of __clifter_slam.Pointclouds__ object using ICP ( iterative closest point ). uses LM ( Levenberg- marquardt ) solver.

```python
provide(
  maps_pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
  frames_pointclouds: clifter_slam.structures.pointclouds.Pointclouds
) -> torch.Tensor
```
uses ICP to compute the relative homogeneous transformation that, when applied to __frams_pointclouds__, would cause the point to alogn with ponts of __maps_pointclouds__.

- parameter
  
  - maps_pointclouds(``clifter_slam.POintclouds``)
    
    object containing batch of matp pointclouds of batch size (B)

  - frames_pointclouds(``clifter_slam.Pointclouds``)

    object containing batch of live frame pointclouds of batch size (B)

- returns
  
  the relative transformation that would align maps_pointclouds with __frames_pointcoulds__

- return type

  ``torch.Tensor``

## clifter_slam.odometry.icputils

```python
solve_linear_system(
  A: torch.Tensor,
  b: torch.Tensor,
  damp: Union[float, torch.Tensor] = 1e-08
)
```
solves the normal equatitaions of linear system Ax = b, fiven the constraint matrix A and the coeeficient vector b. Note that this solves the normal equations, not the linear system. That is, solvet A^t Ax = A^tb, not Ax = b

- parameter
  
  - A (``torch.Tensor``)
    
    the constraint matrix of the linear system
  
  - B (``torch.Tensor``)
  
    The coefficient vector of the linear system

  - damp (``float`` or ``torch.Tensor``)
  
    damping coeeficient to optionally condition the linear system (in practice, a damping coefficient of of P means that we are solving the modified linear system that adds a tiny p to each diagonal element of the constraint matrix A, so that the linear system becomes (A^t A+ __p__L)x) == b, wher I is the identitiy matrix of shape (num_of_variables, num_of_variables)). default : 1e-8

- returns
  
  solution vector of the normal equations of the linear system

- return type

  ``torch.Tensor``

```python
gauss_newton_solve(
  src_pc: torch.Tensor,
  tgt_pc: torch.Tensor,
  tgt_normals: torch.Tensor,
  dist_thresh: Optional[Union[float, int]] = None
)
```
computes gauus newton step by forming linear equation. POints from __src_pc__ which have a distance greater than __dist_thresh__ to the closest point in __tgt_pc__ will be filtered 

- parameters
  
  - src_pc(``torch.Tensor``)
    
    source pointcloud (the pointcloud that needs warping)

  - tgt_pc(``torch.Tensor``)
    
    target pointcloud ( the pointcloud to which the source pointcloud must be warped to)

  - tgt_normals(``torch.Tensor``)

    per-point normal vectors afor each point in the target pointcloud

  - dist_thresh(``float`` or ``int`` or ``None``)

    distance threshold for removing __src_pc__ points distant from tgt_pc. default: None

- return

  tuple containing

  - a (``torch.Tensor``) = linear system equation
  - b (``torch.Tensor``) = linear system residual

- return type

  ``tuple``

```python
point_to_pane_ICP(
  src_pc: torch.Tensor,
  tgt_pc: torch.Tensor,
  tgt_normals: torch.Tensor,
  initial_transform: Optional[torch.Tensor] = None,
  numiters: int = 20,
  damp: float = 1e-08,
  dist_thresh: Optional[Union[float, int]] = None
)
```

computes a rigid transformation between tgt_pc (target_pointcloud) and src_pc ( source pointcloud) using a point-to-plane error metric and the LM (levenberg-marquardt) solver

- parameter
  
  - src_pc(``torch.Tensor``)
    
    source pointcloud (the pointcloud that needs warping)

  - tgt_pc(``toch.Tensor``)

    target pointcloud (the pointcloud to which the source pointcloud must be warped to)

  - tgt_normals(``torch.Tensor``)

    per-point normal vectors for each point in the target pointcloud

  - initial_transform(``torch.Tensor`` or ``None``)

    the initial estimate of the transformation between 'src_pc' and 'tgt_pc'.if None, will use the identity matrix as the initial transfrom. default None

  - numiter(``int``)

    number of iterations to run the optimization for. default 20

  - damp(``float``)

    damping coefficient for nonlinear least-squares. default 1e-8

  - dist_thresh(``float`` or ``int`` or ``None``)

    distance threshold for removing __src_pc__ points distant from tgt_pc. Default None


- return

  tuple containing

  - tansfrom(torch.Tensor) = linear system residual
  - chamfer_indices(torch.Tensor) = Index of the closest point in tgtpc for each point in the src_pc that was not filtered out

- return type

  ``tuple``

```python
point_to_plane_gradICP(
  src_pc: torch.Tensor,
  tgt_pc: torch.Tensor,
  tgt_normals: torch.Tensor,
  initial_transform: OPtional[torch.tensor] = None,
  Numiters: int = 20,
  damp: float = 1e-08,
  dist_thres: Optional[Union[float, int]] = None
  lambda_max: Union[float, int] = 2.0,
  B: Union[float, int] = 1.0,
  B2: Union[float, int] = 1.0,
  nu: Union[float, int] = 200.0
)
```
computes a rigid transformation between tgt_pc (target pointcloud) and src_pc using a point-to-plane error metric and clifter slam LM solver

- parameters
  
  - src_pc(``torch.Tensor``)

    source pointcloud (the pointcloud that needs warping)

  - tgt_pc(``torch.Tensor``)
    
    target pointcloud (the pointcloud to which the source pointcloud must be warped to)

  - tgt_normals(``torch.Tensor``)
    
    per-point normal vectors for each point in the target

  - initial_transform(``torch.Tensor`` or ``None``)
  
    the initial estimate of the transformation between 'src_pc' and 'tgt_pc'. if None, will use the identity matrix as the initial transfrom. default none

  - numiters(``int``)

    number of iterations to run optimization for .default 20

  - damp(``float``)

    damping coefficient for nonlinear least-squares. default 1e-8

  - dist_thres(``float`` or ``int`` or ``None``)

    distance threshold for removing sec_pc points distant from tgt_pc default None

  - lambda_max(``float`` or ``int``)

    maximum value the damping function can assume (lambda_min will be ``1/mabda_max``)

  - B (``float`` or ``int``)

    clifter_slam fallof control parameter

  - B2 (``float`` or ``int``)

    clifter_slam control paramater

  - nu (``float`` or ``int``)

    clifter_slam control parameter

- returns

  tuple containing 

  - tansformation (torch.Tensor) = linear system residual
  - chamfer_indices (torch.Tensor) = index of the closest point in tgt_pc for each point in src_pc that was not filtered out

- return type
  
  ``tuple``

```python
downsample_pointclouds(
  pointclouds: clifter_slam.structures.pointclouds.Pointclouds,
  pc2im_bnhw: torch.Tensor,
  ds_ratio: int
) -> clifter_slam.structures.pointclouds.Pointclouds
```
downsamples active points of pointclouds (points that porject inside the live frame) and removes non-active points

- parameters
  
  - pointclouds (clifter_slam.Pointclouds)

    pointclouds to downsample

  - pc2im_bnhw (torch.Tensor)

    active map points lookup table. each row contains batch index b, point index (in pointclouds) n, and height and width index after projection to live frame h aand w respectively

  - ds_ratio(``int``)

    downsampling ratio

- returns 

  downsampling pointclouds

- return type

  clifter_slam.pointclouds


```python
downsample_rgbdimages(
  rgbdimages: clifter_slam.structures.rgbdimages.RGBDImages,
  ds_ratio: int,
) -> clifter_slam.structures.pointclouds.Pointclouds
```
downsample points and normals of RGBDImages and retruns a clifter_slam.Pointclouds object

- parameters
  
  - rgbdimages (clifter_slam.RGBDImges)

    RGBDImages to downsample

  - ds_ratio(``int``)

    downsampling ratio

- returns
  
  downsampled points and normals

- return type
  
  clifter_slam.Pointclouds

