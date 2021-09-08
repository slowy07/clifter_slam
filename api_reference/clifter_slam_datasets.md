# clifter_slam.datasets

## clifter_slam.datasets.icl

[source file clifter_slam.datasets.icl](https://github.com/slowy07/clifter_slam/blob/main/clifter_slam/datasets/icl.py)

torch dataset for loading in th ICL NUIM dataset. will fetch sequences of rgb images, depth maps, intrinsics matrix, poses, frame to frame relative transformations (with first frame's pose as the reference transformation), names of frames. expects the following folder structure for the ICL dataset
```
| ├── ICL
| │   ├── living_room_traj0_frei_png
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── associations.txt
| │   │   └── livingRoom0n.gt.sim
| │   ├── living_room_traj1_frei_png
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── associations.txt
| │   │   └── livingRoom1n.gt.sim
| │   ├── living_room_traj2_frei_png
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── associations.txt
| │   │   └── livingRoom2n.gt.sim
| │   ├── living_room_traj3_frei_png
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── associations.txt
| │   │   └── livingRoom3n.gt.sim
| │   ├── living_room_trajX_frei_png
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── associations.txt
| │   │   └── livingRoomXn.gt.sim
|
```
example of sequence creation from frames with ``seqlen =4``, ``dilation=1``,``stride=3`` and ``start=2``
```
                                    sequence0
                ┎───────────────┲───────────────┲───────────────┒
                |               |               |               |
frame0  frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8  frame9  frame10  frame11 ...
                                        |               |               |                |
                                        └───────────────┵───────────────┵────────────────┚
                                                            sequence1
```

with parameters

- basedir

    path to the base directory containing the ``living_room_trajX_frei_png/`` directories from ICL-NUIM. each trajectory subdirectory is assumed to contain ``depth/``, ``rgb/``, ``assosciations.txt`` and ``livingRoomOn.gt.sim``
    ```
    ├── living_room_trajX_frei_png
    ├── depth/
    ├── rgb/
    ├── associations.txt
    └── livingRoomXn.gt.sim
    ```

- trajectories (``str`` or ``tuple`` of ``str`` or ``None``)

    trajectories to use from
    - ``living_room_traj0_frei_png``
    - ``living_room_traj1_frei_png``
    - ``living_room_traj2_frei_png``
    - ``living_room_traj3_frei_png``
    
    can be path to a ``.txt`` file where each line is a trajectory name

- seqlen(``int``)

    number of rames to use for each sequence of frames. default is ``4``

- dilation(``int`` or ``None``)

    number of (original trajectory's) frames to skip between two consecutive frames in the extracted sequence. see above sexample if unsure. if None,will set ``dilation = 0`` .default ``None``

- stride(``int`` or ``None``)

    number of frames between the first frames of two consecutive extracted sequences. see aboce example if unsure. if ``None``, will set ``stride = seqlen * (dilation + 1)`` (non-overlapping sequences). default ``None``

- start(``int`` or ``None``)

    index of the frame from which to start extracting sequences for every trajectory. if ``None``, will start from the first frame. Default ``None``

- end(``int``)

    index of the frame at which to stop extracting sequences fro every trajectory. if ``None``, will continue extracting frames until the end of the trajectory. Default ``None``

- height(``int``)

    spatial height to resize frames to. Default ``480``

- width(``int``)

    spatial width to resize frames to. Default ``640``

- channels_first(``bool``)

    if ``True``, will use channels first representation (B, L, C, H, W) for images(``batchsize``, ``sequencelength``, ``channles``, ``height``, ``width``) if ``False``, channels last representation (B, L, H, W, C) default ``False``

- normalize_color(``bool``)

    normalize color to range ``[01]`` or leave it at range ``[0255]``. Default ``False``.

- return_depth(``bool``)

    determines whether to return depths. Default ``True``

- return_pose(``bool``)

    determines wheteher to return intinsics. Default ``True``

- return_transform(``bool``)

    determines wheter to return transforms w.r.t initial pose being tansformed to be identity. Default ``True``
    
- return_names(``bool``)

    deterimnes whether to return sequences names. Default ``True``


examples:

```
>>> dataset = ICL(
    basedir="ICL-data/",
    trajectories=("living_room_traj0_frei_png", "living_room_traj1_frei_png")
    )
>>> loader = data.DataLoader(dataset=dataset, batch_size=4)
>>> colors, depths, intrinsics, poses, transforms, names = next(iter(loader))
```

## clifter_slam.datasets.scannet

[source file clifter_slam.datasets.scannet](https://github.com/slowy07/clifter_slam/blob/main/clifter_slam/datasets/scannet.py)

a torch dataset for loading in ``Scannet dataset`` will fetch sequences of rgb images, depth maps, intrinsics matices, poses, frame to frame relative transformations (with first frame's pose as the reference transformation), names of sequences, and semantic segmentation labels

parameters

- basedir(``str``)

    path to the base directory containing the ``sceneXXXXX_XX/`` directorues from ScanNet. Each secene subdirectoryis assumed to contain ``color/``, ``depth/``, ``intrinsic/``, ``label-fit/``, and ``pose/`` directories

- seqmetadir(``str``)

    path to directory containing sequence associations. directory is assumed to contain metadata ``.txt`` files (one metadata per sequence)

- scenes(``str`` or ``tuple`` of ``str``)

    scenes to use from sequences (used for creating train/val/test split) can be path to a ``.txt`` file where each line is a scene name (``sceneXXXX_XX``), atuple of scene names, or, None to use all scenes

- start(``int``)

    index of the frame from which to start for every sequence. Default ``0``

- end(``int``)

    index of the frame at which to end every sequence. Defailt ``-1``

- height(``int``)

    spatial height to resize frames to. Default ``480``

- width(``int``)

    spatial width to resize frames to. Default ``640``

- seg_classes(``str``)

    the pallete classes that the network should learn. either ``nyu40`` or ``scannet20``. default ``scannet20``

- channels_first(``bool``)

    if ``True``, will use channles first representation (B, L, C, H, W) for images (``batchsize``, ``sequencelength``, ``channels``, ``height``, ``width``). if ``False``, will use channels last representation (B, L, H, W, C) default ``False``

- normalize_color(``bool``)

    normalize color to range``[0, 1]`` or leave it at range ``[0,255]`` default ``False``

- return_depth(``bool``) 

    determines wheteher to return depth default ``True``

- return_intrinsics(``bool``)

    determines whether to return intrinsics. default ``True``

- return_pose(``bool``)

    determines whether to return poses. default ``True``

- return_transform(``bool``)

    determines whether to return transforms w.r.t, initial pose being transformed to be identity. default ``True``

- return_names(``bool``)

    deterimines whether to return sequence names. Default ``True``

- return_labels(``bool``)

    determines whether to return segmentation labels. default ``True``


examples
```
>>> dataset = Scannet(
    basedir="ScanNet-gradSLAM/extractions/scans/",
    seqmetadir="ScanNet-gradSLAM/extractions/sequence_associations/",
    scenes=("scene0000_00", "scene0001_00")
    )
>>> loader = data.DataLoader(dataset=dataset, batch_size=4)
>>> colors, depths, intrinsics, poses, transforms, names, labels = next(iter(loader))
```

## clifter_slam.datasets.tum

a torch dataset for loading in the ``TUM`` dataset. will fetch sequences of rgb images, depth maps, intrinsics matrix, poses, frame to frame relative transformations (with first frame's pose as the reference transformation), names of frames. uses extacted ``.tgz`` sequences download from [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/download). expects similar to the following folder structure for the TUM dataset

```
| ├── TUM
| │   ├── rgbd_dataset_freiburg1_rpy
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── accelerometer.txt
| │   │   ├── depth.txt
| │   │   ├── groundtruth.txt
| │   │   └── rgb.txt
| │   ├── rgbd_dataset_freiburg1_xyz
| │   │   ├── depth/
| │   │   ├── rgb/
| │   │   ├── accelerometer.txt
| │   │   ├── depth.txt
| │   │   ├── groundtruth.txt
| │   │   └── rgb.txt
| │   ├── ...
|
|
```

example of sequence create from frames with ``seqlen=4``, ``dilation = 1``, ``stride = 3`` and ``start = 2``

```
                                    sequence0
                ┎───────────────┲───────────────┲───────────────┒
                |               |               |               |
frame0  frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8  frame9  frame10  frame11 ...
                                        |               |               |                |
                                        └───────────────┵───────────────┵────────────────┚
                                                            sequence1
```

parameters

- basedir(``str``)

    path to the base directory containing extracted TUM sequences in separate directorues. each sequence subdirectory is assumed to contain ``depth``, ``rgb/``, ``accelerometer.txt``, ``depth.txt`` and ``groundtruth.txt`` and ``rgb.txt`` E.g:

    ```
    ├── rgbd_dataset_freiburgX_NAME
    ├── depth/
    ├── rgb/
    ├── accelerometer.txt
    ├── depth.txt
    ├── groundtruth.txt
    └── rgb.txt
    ```

- sequences(``str`` or ``tuple`` of ``str`` or ``None``)

    sequence to use from those available in ``basedir``, can be path to a ``.txt`` file where each line is asequence name (e.g ``rgb_dataset_freiburg1_rpy``), a tuple of sequence names, or None to use all sequence. Default ``None``

- seqlen(``int``)

    number of frames to use for each sequence of frames. Default ``4``

- dilation(``int`` or ``None``)

    number of (original trajectory's) frames to skip between two consecutive frame in the extracted sequence. see above example is unsure. if ``None``, will set ``dilation = ``, Default ``None``

- start(``int`` or ``None``)

    index of rgb frame from which to start extracting sequneces for every sequence. if ``None``, will start from the first frame. Default ``None``

- end(``int``)

    index of the rgb frame at which to stop extracting sequences for every sequence. if ``None``, will continue extracting frames, untul the end of the sequence. Default ``None``

- height(``int``)

    spatial height to reseize frames to. default ``480``

- width(``int``)

    spatial width to resize frames to. Default ``640``

- channels_first(``bool``)

    if ``True``, will use channels first representation (B, L, C, H, W) for images (``bathsize``, ``sequencelength``, ``channels``, ``height``, ``width``) if false, will use channels last representation (B, L, H, W, C) default ``False``

- nomalize_color(``bool``)

    normalize color to range ``[01]`` or leave it at range ``[0255]``.default ``False``

- return_depth(``bool``)

    determines whether to return depths. default ``True``

- return_intrinsics(``bool``)

    determines whether to return intrinsics. default ``True``

- return_pose(``bool``)

    determines whether to return poses. default ``True``

- return_transform(``bool``)

    determines wheter to return transform w.r.t initial pose being transformed to be identity. default ``True``

- return_names(``bool``)

    determines whether to return sequence names. default ``True``

- return_timestamps(``bool``)

    determines whether to return  rgb, depth and pose timestamp default ``True``

examples
```
>>> dataset = TUM(
    basedir="TUM-data/",
    sequences=("rgbd_dataset_freiburg1_rpy", "rgbd_dataset_freiburg1_xyz"))
>>> loader = data.DataLoader(dataset=dataset, batch_size=4)
>>> colors, depths, intrinsics, poses, transforms, names = next(iter(loader))
```

## clifter_slam.datasets.datautils

```python
normalize_image(rgb: Union[torch.Tensor, numpy.ndarray])
```
normalizes RGB image value from ``[0, 255]`` range to ``[0, 1]`` range

parameter

- **rgb**(``torch.Tensor`` or ``numpy.ndarray``)
    
    RGB image in range ``[0, 255]``

- **Returns**

    normalize RGB image in range ``[0, 1]``

- **Return type**

    ``torch.Tensor`` or ``numpy.ndarray``

- **shape**

    - ``rgb`` (*) (any shape)
    - output: same shape as input (*)

```python
channels_first(rgb: Union[torch.Tensor, numpy.ndarray])
```
convert from channels last representation (*, H, W, C) to channels first representation (, C, H, W)

parameters

- **rgb**(``torch.Tensor`` or ``numpy.ndarray``)

    (*, H, W, C) ordering (,height, width, channles)

- **Returns**

    (*, C, H, W) ordering

- **Return type**

    ``torch.Tensor`` or ``numpy.ndarray``

- **shape**

    - rgb (*, H, W, C)
    - Output (*, C, H, W)

```python
scale_intrinsics(
    intrinsics: Union[numpy.ndarray, torch.Tensor],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int]
)
```
scales the intrinsics appropriately for resized frames where Hratio = Hnew / Hold and Wratio = Wnew / wold

**parameters**

- intrinsics(``numpy.ndarray`` or ``torch.Tensor``)

    intrinsics matrix of original frame

- h_ratio(``float`` or ``int``)

    ratio of new frame's height to old frame's height H``ratio`` = H``new`` / H``old``

- w_ratio(``float`` or ``int``)

    ratio of new frame's width to old frame's width W``ratio`` = W``new`` / W``old``

**Returns**

intrinsics matrix scaled approprately for new frame size

**Return type**

``numpy.ndarray`` or ``torch.Tensor``

**shape**

- intrinsics (*, 3, 3) or (, 4, 4)
- output: matches ``intrinsics`` shape (*,3, 3) or (, 4, 4)


```python
pointquanternion_to_homogeneous(
    pointquaternions: Union[numpy.ndarray, torch.Tensor],
    eps: float = 1e-12
)
```
convert 3D point and unit quaternions (Tx, Ty, Tz, Qx, Qy, Qz, Qw) to homogeneous transformations ``[R | t]`` where ``R`` denotes the (3, 3) rotation matrix and ``T`` denotes the (3, 1) translation matrix

|R      |T      |
|:---   |   ---:|
|0 0 0  |     1 |

parameters

- pointquaternions(``numpy.ndarray`` or ``torch.Tensor``)

    3D point positions and unit quaternions (Tx, Ty, Tz, Qx, Qy, Qz, Qw) where (Tx, Ty, Tz) is the 3D position and (Qx, Qy, Qz, Qw) is the unit quternion

- returns
    
    Homogeneous transformations matrices

- return type

    ``numpy.ndarray`` or ``torch.Tensor``

- shape

    - pointquaternions (*, 7)
    - output (*, 4, 4)

```python
poses_to_transforms(
    poses: Union[numpy.ndarray],
    List[numpy.ndarray]
)
```
parameters

- poses(``nump.ndarray`` or ``list`` of ``numpy.ndarray``)

    sequence of poses in ``numpy.ndarray`` format

return

- sequence of frame to frame transformations where initial

    frame is transformed to have identity pose
    type: ``numpy.ndarray``

```python
create_label_image(
    prediction: numpy.ndarray,
    color_palette: collections.OrderedDict
)
```

creates a label image, given a network prediction (each pixel contains class index) and a color pallete

parameters

- prediction(``numpy.ndarray``)

    predicted image where each pixel contains an integer, corresponding to its class label

- color_pallete(``OrderedDict``)

    contains RGB colors (uint8) for each class

returns

label image wiht the given color pallete