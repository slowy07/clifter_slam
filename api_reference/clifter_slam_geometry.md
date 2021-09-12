# clifter_slam.geometry


```python
homogenize_points(pts: torch.Tensor)
```
convert a set of points to homogeneous coordinates

- parameters
  
  ``pts(torch.Tensor)``

  tensor containing points to be homogenized

- returns
  
  homogeneous coordinates of pts

- return type

  ``torch.Tensor``

examples:
```
>>> pts = torch.rand(10, 3)
>>> pts_homo = homogenize_points(pts)
>>> pts_homo.shape
torch.Size([10, 4])
```

```python
project_points(aclam_coords: torch.Tensor, proj_mat: torch.Tensor, eps: Optional[float] = 1e-06) -> torch.Tensor
```

project points from the camera coordinate frame to the image (pixel) frame

- parameters
  
  - **cam_coords**(``torch.Tensor``)
    
    piexl coordinates (defined in the frame of the first camera)

  - returns

    image (pixel) coordinates corresponding to the input 3D points
  
  - return Type

    ``torch.Tensor``
  
example
```
>>> cam_coords = torch.rand(10, 4)
>>> proj_mat = torch.rand(4, 4)
>>> pixel_coords = project_points(cam_coords, proj_mat)
>>> pixel_coords.shape
torch.Size([10, 2])
```

```python
unproject_points(pixel_coords: torch.Tensor, intrinsic_inv: torch.Tensor, depths: torch.Tensor) -> torch.Tensor
```
unproject points from the image (pixel) frame to the camera coordinate frame

- parameters
  - ``pixel_coords``(``torch.Tensor``)

    pixel coordinates
  
  - ``intrinsic_inv``(``torch.Tensor``)

    inverse of the intrinsic matrix
  
  - ``depths``(``torch.Tensor``)

    per pixel depth estimates

- returns

  camera coordinates

- returns type

  ``torch.Tensor``

examples
```
>>> pixel_coords = torch.rand(10, 3)
>>> intrinsic_inv = torch.rand(3, 3)
>>> depths = torch.rand(10)
>>> cam_coords = unproject_points(pixel_coords, intrinsic_inv, depths)
>>> cam_coords.shape
torch.Size([10, 3])
```

```python
inverse_intrinsics(K: torch.Tensor, eps: float = 1e-06) -> torch.Tensor
```
Efficient inversion of intrinsics matrix

- paramaeters
  - K (``torch.Tensor``)
    
    intrinsics matrix
  - eps (``float``)

    Epsilon for numerical stability
  
- returns

  ``torch.Tensor``
