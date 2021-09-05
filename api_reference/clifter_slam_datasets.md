# clifter_slam.datasets.icl

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

