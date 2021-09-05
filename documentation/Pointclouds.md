## Pointclouds

``Pointclouds`` structures aim to contain batched pointclouds and allow for batched operation on pointclouds.

``Pointclouds`` object can be initialized from points coordinates, point normals, point colors and (optionally) point features. these attributes can be passed in one of the following representation
- list (of ``torch.Tensor``) store points of each pointcloud of shape (Nb,3) in a list of B ``torch.Tensor`` objects
- padded: store all points in a (B,N,3) tensor

``Pointclouds`` can also be intatiated from ``RGBDImages``


```python
from clifter_slam.structures.utils import pointclouds_from_rgbdimages

# instantiate empty Pointclouds object
pointclouds = Pointclouds()
print(pointclouds.has_points)  # False
print('---')

# instantiation from list of tensors of points
pointclouds = Pointclouds(points=[torch.rand(4, 3), torch.rand(2, 3), torch.rand(1, 3)],
                          normals=[torch.rand(4, 3), torch.rand(2, 3), torch.rand(1, 3)],
                          colors=[torch.rand(4, 3), torch.rand(2, 3), torch.rand(1, 3)])
print(pointclouds.num_points_per_pointcloud)  # tensor([4, 2, 1])
print('---')

# instantiation from tensor
pointclouds = Pointclouds(points=torch.rand(3, 4, 3),
                          normals=torch.rand(3, 4, 3),
                          colors=torch.rand(3, 4, 3))
print(pointclouds.num_points_per_pointcloud)  # tensor([4, 4, 4])
print('---')

# instantiate with features
# features can have any number of dimensions
pointclouds = Pointclouds(points=torch.rand(3, 4, 3),
                          normals=torch.rand(3, 4, 3),
                          colors=torch.rand(3, 4, 3),
                          features=torch.rand(3, 4, 10))
print(pointclouds.has_features)  # True
print('---')

# instantiate from RGBDImages with sequence length of 1
rgbdimages1 = rgbdimages[:, 0]
pointclouds = pointclouds_from_rgbdimages(rgbdimages1, filter_missing_depths=False)
print(rgbdimages1.shape)  # (2, 1, 240, 320)
print(pointclouds.num_points_per_pointcloud)  # tensor([76800, 76800])
```
with output
```
False
---
tensor([4, 2, 1])
---
tensor([4, 4, 4])
---
True
---
(2, 1, 240, 320)
tensor([76800, 76800])
```

## list and padded internal representations
similiar to Pytorch3d, our ``pointclouds`` structure suppors a list representation and a padded representation internally.
```python
torch.manual_seed(0)

# instantiation from list of tensors of points
pointclouds = Pointclouds(points=[torch.rand(2, 3), torch.rand(1, 3)],
                          normals=[torch.rand(2, 3), torch.rand(1, 3)],
                          colors=[torch.rand(2, 3), torch.rand(1, 3)],
                          features=[torch.rand(2, 10), torch.rand(1, 10)])
print(pointclouds.num_points_per_pointcloud)  # tensor([2, 1])
print('---')

# List representation
for points in pointclouds.points_list:
    print(points)
# tensor([[0.4963, 0.7682, 0.0885],
#         [0.1320, 0.3074, 0.6341]])
# tensor([[0.4901, 0.8964, 0.4556]])
print('---')

# Padded representation
print(pointclouds.points_padded)
# tensor([[[0.4963, 0.7682, 0.0885],
#          [0.1320, 0.3074, 0.6341]],

#         [[0.4901, 0.8964, 0.4556],
#          [0.0000, 0.0000, 0.0000]]])
print('---')

# Padded representation shapes
print(pointclouds.points_padded.shape)  # torch.Size([2, 2, 3])
print(pointclouds.normals_padded.shape)  # torch.Size([2, 2, 3])
print(pointclouds.colors_padded.shape)  # torch.Size([2, 2, 3])
print(pointclouds.features_padded.shape)  # torch.Size([2, 2, 10])
print('---')

# List representation shapes
print([p.shape for p in pointclouds.points_list])  # [torch.Size([2, 3]), torch.Size([1, 3])]
print([n.shape for n in pointclouds.normals_list])  # [torch.Size([2, 3]), torch.Size([1, 3])]
print([c.shape for c in pointclouds.colors_list])  # [torch.Size([2, 3]), torch.Size([1, 3])]
print([f.shape for f in pointclouds.features_list])  # [torch.Size([2, 10]), torch.Size([1, 10])]
```
with output
```
tensor([2, 1])
---
tensor([[0.4963, 0.7682, 0.0885],
        [0.1320, 0.3074, 0.6341]])
tensor([[0.4901, 0.8964, 0.4556]])
---
tensor([[[0.4963, 0.7682, 0.0885],
         [0.1320, 0.3074, 0.6341]],

        [[0.4901, 0.8964, 0.4556],
         [0.0000, 0.0000, 0.0000]]])
---
torch.Size([2, 2, 3])
torch.Size([2, 2, 3])
torch.Size([2, 2, 3])
torch.Size([2, 2, 10])
---
[torch.Size([2, 3]), torch.Size([1, 3])]
[torch.Size([2, 3]), torch.Size([1, 3])]
[torch.Size([2, 3]), torch.Size([1, 3])]
[torch.Size([2, 10]), torch.Size([1, 10])]
```

## indexing and slicing
basic indexing and slicing of ``Pointclouds`` over the first (batch) dimension is supported.
```python
# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand(3, 4, 3),
                          normals=torch.rand(3, 4, 3),
                          colors=torch.rand(3, 4, 3))
print(len(pointclouds))  # 3
print('---')

# indexing
pointclouds1 = pointclouds[0]
print(len(pointclouds1))  # 1
print('---')

# slicing
pointclouds2 = pointclouds[:2]
print(len(pointclouds2))  # 2
```
with output
```
3
---
1
---
2
```

## translations, rotations and transformations
``Pointclouds`` support batch mode geometric operations such as translations, rotations, and transformations
```python
import plotly.graph_objects as go
torch.manual_seed(0)

def custom_plotly_viz(pointclouds, title):
    fig = go.Figure(pointclouds.plotly(0, as_figure=False, point_size=15))
    fig.update_layout(title=title, autosize=False, height=400, width=400)
    fig.show()

# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand(2, 5, 3),
                          normals=torch.rand(2, 5, 3),
                          colors=torch.rand(2, 5, 3))

# translate
pointclouds1 = pointclouds + 10

# scale
pointclouds2 = pointclouds * 100

# rotate (Bx3x3 rotation)
rmat = torch.tensor(
    [
     [(3 ** 0.5) / 2, -0.5, 0],
     [0.5, (3 ** 0.5) / 2, 0],
     [0, 0, 1],
     ]
     )
pointclouds3 = pointclouds.rotate(rmat)

# transform (Bx4x4 transformation)
mat = torch.tensor(
    [
     [(3 ** 0.5) / 2, -0.5, 0, 20],
     [0.5, (3 ** 0.5) / 2, 0, 20],
     [0, 0, 1, 20],
     [0, 0, 0, 1],
     ]
     )
pointclouds4 = pointclouds.transform(mat)

# visualizations
custom_plotly_viz(pointclouds, "pointclouds[0]")
custom_plotly_viz(pointclouds1, "pointclouds[0] + 10")
custom_plotly_viz(pointclouds2, "pointclouds[0] * 100")
custom_plotly_viz(pointclouds3, "pointclouds[0] rotated 30 deg about z-axis")
custom_plotly_viz(pointclouds4, "pointclouds[0] rigid transformation")
```

## pinhole camera projection
``Pointclouds`` can be projected into a 2-d plane given the intrinsics matrix using the ``Pointclouds.pinhole_projection(instrinsics)``
```python
import plotly.graph_objects as go
torch.manual_seed(0)

def custom_plotly_viz(pointclouds, title):
    fig = go.Figure(pointclouds.plotly(0, as_figure=False, point_size=15))
    fig.update_layout(title=title, autosize=False, height=400, width=400)
    fig.show()

# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand(2, 5, 3),
                          normals=torch.rand(2, 5, 3),
                          colors=torch.rand(2, 5, 3))

# pinhole projection
fx, fy = 1., 1.
cx, cy = 0.5, 0.5
intrinsics = torch.tensor(
    [
     [fx, 0, cx, 0],
     [0, fy, cy, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]
    ]
)
pointclouds1 = pointclouds.pinhole_projection(intrinsics)

# visualizations
custom_plotly_viz(pointclouds, "pointclouds[0]")
custom_plotly_viz(pointclouds1, "pointclouds[0] projection")
```

## transfer between GPU / CPU
``Pointclouds`` support easy transfer between CPU and GPU. this operation transfers all tensors in the ``PointClouds`` object between CPU / GPU

```python
# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand(2, 5, 3),
                          normals=torch.rand(2, 5, 3),
                          colors=torch.rand(2, 5, 3))

# transfer to GPU
if torch.cuda.is_available():
    pointclouds = pointclouds.to("cuda")
    pointclouds = pointclouds.cuda()  # equivalent to pointclouds.to("cuda")
    print(pointclouds.points_padded.device)  # "cuda:0"
    print('---')

# transfer to CPU
pointclouds = pointclouds.to("cpu")
pointclouds = pointclouds.cpu()  # equivalent to pointclouds.to("cpu")
print(pointclouds.points_padded.device)  # "cpu"
```
with output
```
cuda:0
---
cpu
```

## detach and clone tensors
``Pointclouds.detach`` returns aa new ``Pointclouds`` object such that all internal tensor of the new object. ``Pointclouds.clone()`` returns a new ``Pointclouds`` object such that aall the internal tensor are cloned
```python
# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand((2, 5, 3), requires_grad=True),
                          normals=torch.rand((2, 5, 3), requires_grad=True),
                          colors=torch.rand((2, 5, 3), requires_grad=True))

# clone
pointclouds1 = pointclouds.clone()
print(torch.allclose(pointclouds1.points_padded, pointclouds.points_padded))  # True
print(pointclouds1.points_padded is pointclouds.points_padded)  # False
print('---')

# detach
pointclouds2 = pointclouds.detach()
print(pointclouds.points_padded.requires_grad)  # True
print(pointclouds2.points_padded.requires_grad)  # False
```
with output
```
True
False
---
True
False
```

## visualization
``Pointclouds`` can quickly and easily be visualized with either the ``.plotly(batch_index)`` method or the ``.open3d(batch_index)``
```python
import open3d as o3d
torch.manual_seed(0)

# initalize Pointclouds
pointclouds = Pointclouds(points=torch.rand((2, 5, 3), requires_grad=True),
                          normals=torch.rand((2, 5, 3), requires_grad=True),
                          colors=torch.rand((2, 5, 3), requires_grad=True))

# plotly visualization
pointclouds.plotly(0, point_size=20).show()

# open3d visualization (does not work with Google Colab)
# o3d.visualization.draw_geometries([pointclouds.open3d(0)])
```