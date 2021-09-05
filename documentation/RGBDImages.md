# RGBDImages

the ``RGBDImages`` structures aims to contain batched frametensors to more easily pass on to SLAM algorithms.It also supports easy computation of (both local and global) vertex maps and normal maps

an ``RGBDImages`` object acan be initialized from rgb images, depth images, instrinsics and (optionally) poses.``RGBDImages`` supports both a channels first and a channels last representation.

```python
print(f"colors shape: {colors.shape}")  # torch.Size([2, 8, 240, 320, 3])
print(f"depths shape: {depths.shape}")  # torch.Size([2, 8, 240, 320, 1])
print(f"intrinsics shape: {intrinsics.shape}")  # torch.Size([2, 1, 4, 4])
print(f"poses shape: {poses.shape}")  # torch.Size([2, 8, 4, 4])
print('---')

# instantiation without poses
rgbdimages = RGBDImages(colors, depths, intrinsics)
print(rgbdimages.shape)  # (2, 8, 240, 320)
print(rgbdimages.poses)  # None
print('---')

# instantiation with poses
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
print(rgbdimages.shape)  # (2, 8, 240, 320)
print('---')
```
with ouput
```
colors shape: torch.Size([2, 8, 240, 320, 3])
depths shape: torch.Size([2, 8, 240, 320, 1])
intrinsics shape: torch.Size([2, 1, 4, 4])
poses shape: torch.Size([2, 8, 4, 4])
---
(2, 8, 240, 320)
None
---
(2, 8, 240, 320)
---
```

## indexing and slicing

basic indexing and slicing of ``RGBDImages`` over the first(batch) dimesion and the second (sequence length) dimension is supported.

```python
# initalize RGBDImages
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)

# indexing
rgbdimages0 = rgbdimages[0, 0]
print(rgbdimages0.shape)  # (1, 1, 240, 320)
print('---')

# slicing
rgbdimages1 = rgbdimages[:2, :5]
print(rgbdimages1.shape)  # (2, 5, 240, 320)
print('---')
```
with output
```
(1, 1, 240, 320)
---
(2, 5, 240, 320)
---
```

## vertex maps and normal maps
this section demonstrates accessing vertex maps and normal maps from ``RGBDImages``. vertex maps are computed when accessing the ``RGBDImages.vertex_maps`` property, and are cached afterwards for additional aaccess without further computation (and similarly with normal maps).

``RGBDImages``has both a local vertex map property(``RGBDImages.vertex_map``) which computes vertex position with respect to each frame, as well as global vertex map (``RGBDImages.global_vertex_map``) which considers the poses of the ``RGBDImages`` object to compute the global vertex positions. A similiar story is true for ``RGBDImages.normal_map`` and ``RGBDImages.global_normal_map``

## transfer between GPU/CPU

``RGBDImages`` support easy transfer between CPU and GPU. this operation transfers all tensors in the ``RGBDImages`` objects between CPU/GPU
```python
# initalize RGBDImages
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)

if torch.cuda.is_available():
    # transfer to GPU
    rgbdimages = rgbdimages.to("cuda")
    rgbdimages = rgbdimages.cuda()  # equivalent to rgbdimages.to("cuda")
    print(rgbdimages.rgb_image.device)  # "cuda:0"
    print('---')

# transfer to CPU
rgbdimages = rgbdimages.to("cpu")
rgbdimages = rgbdimages.cpu()  # equivalent to rgbdimages.to("cpu")
print(rgbdimages.rgb_image.device)  # "cpu"
```
with the output
```
cuda:0
---
cpu
```

## detach and clone tensors
``RGBDImages.detach`` returns a new ``RGBDImages`` object such that all internal tensors of the new object do not require grad. ``RGBDImages.clone()`` returns a new ``RGBDImages`` object such that all internal tensors are cloned.

```python
# initalize RGBDImages
rgbdimages = RGBDImages(colors.requires_grad_(True),
                        depths.requires_grad_(True),
                        intrinsics.requires_grad_(True),
                        poses.requires_grad_(True))

# clone
rgbdimages1 = rgbdimages.clone()
print(torch.allclose(rgbdimages1.rgb_image, rgbdimages.rgb_image))  # True
print(rgbdimages1.rgb_image is rgbdimages.rgb_image)  # False
print('---')

# detach
rgbdimages2 = rgbdimages.detach()
print(rgbdimages.rgb_image.requires_grad)  # True
print(rgbdimages2.rgb_image.requires_grad)  # False
```
with output
```
True
False
---
True
False
```

## channels first and channels last representation

``RGBDImages`` supports both a channels first and a channles last representation. These representation can be transformed to one another with ``to_channels_first()`` and ``to_channels_last()``

```python
# initalize RGBDImages
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
print(rgbdimages.rgb_image.shape)  # torch.Size([2, 8, 240, 320, 3])
print('---')

# convert to channels first representation
rgbdimages1 = rgbdimages.to_channels_first()
print(rgbdimages1.rgb_image.shape)  # torch.Size([2, 8, 3, 240, 320])
print('---')

# convert to channels last representation
rgbdimages2 = rgbdimages1.to_channels_last()
print(rgbdimages2.rgb_image.shape)  # torch.Size([2, 8, 240, 320, 3])
print('---')
```
with output
```
torch.Size([2, 8, 240, 320, 3])
---
torch.Size([2, 8, 3, 240, 320])
---
torch.Size([2, 8, 240, 320, 3])
---
```

## visualization
for easy and quick visualization of ``RGBDImages``, one can use the ``plotly(batch_index)``
```python
# initalize RGBDImages
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)

# visualize
rgbdimages.plotly(0).show()
```