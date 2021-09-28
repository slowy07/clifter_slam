# clifter slam

clifter_slam is an open-source library aimed at providing implementation of SLAM subsystem for deeplearning (DL) practioners. While several DL practitioners want to use SLAM from the comfort of theri favorite DL library (TensorFlow / Pytorch), there are currentyl no freely available implementations.We aim to fill in that gap with this library

Another underlying motivation is to release code for the clifter_slam paper, where we demonstrate end-to-end differentiable dense SLAM systems. People should be able to load up their favorite datasets and run differentiable / no-differentiable dense SLAM system on them

## SLAM
simlatneous localization and mapping (SLAM) has for decades been a central problem in robot preception and state estimation. A large portion of the SLAM literature has focused either dircetly or indirectly on the question of map representation. This fundamental choice dramatically impacts the choice of processing blocks in the SLAM pipeline, as well as all other downstream task that depend on the output of the SLAM system. of late, gradient based learnaing approacehs have tansformed the outlook of several domains (e.g image recognition, language modelling, speech recognition) however, such techniqueshave had limited success in the context of SLAM, primarily since many of the elements in the standard SLAM system would enable task-driven representation learrning since the error signals indicatig task performance could be back-prpagated all the way through the SLAM system, to the raw sensor observations.

this is particualarly __true__ for __dense__ 3D __maps__ generated from RGB-D cameras, where there has been a lack of consensus on the right representation (pointclouds, meshes, surfels, etc). several methods gave demonstrated a capability for producing dense 3D maps from sequences of RGB or RGB-D frames, however, none of these method are able to solve the __inverse mapping problem__.

## differentiable map fusion
the aforementioned diferntiable mapping strategy, while providing us with a smooth observation model, also causes in propotion with exploration time. However, map elements should ideally increase with propotion to the __explored volume of occupied space__, rather than with exploration time. Convetional dense mapping techniques (e.g kinectfusion, pointfusion) employ this through __fusion__ of reudant observation of the same map element, as a consequence, the recorvered map has a more managable size, but moer impotantly, the reconstruction quality improves greatlt , while most fusion strategies are differentiable, they impose fallof thresholds that cause an abrupt change in gradient flow at the trncation point. we use a logistuc fallof function.

## differentiable optimization

we first design a test suite nonlinear curve fitting probemsm to measure performance of clifter_LM to its non-differentiable conterpart. we consider three nonliner fuctions
```
f(x) = a exp (- (x - t)^ 2 / 2 w ^ 2)
f(x) = sin(ax + tx + w)
f(x) = sinc(ax + tx + w)
```
for each of these functions, we uniformly sampel the paramters __p__ = {a, t, w} to create a suite of ground truth curves, uniformly sample an initial guess p0 = {a0, t0, w0} in interva [-6, 6], we sample 100 problem instance s for each of the three functions. we run a variety of optimizers (such as gradient descent (gd))

## qualintative results

clifter_slam works out of the box on multiple other RGB-D datasets,speciafically, we present qualitative result of running or differentiable SLAM system on RGB-D sequences frin TUM RGB-D dataset, ScanNet as well as on an in-house sequence captured from an Intel Real Sense camera


+------------------------------------+----------+--------+
|               method               |   ATE    |  RPE   |
+------------------------------------+----------+--------+
| ICP-Odometry (non-differentiable)  |   0.029  | 0.0318 |
| ICP-Odometry                       | 0.01664  | 0.0237 |
| ICP-SLAM (non-differentiable)      |  0.0282  | 0.0294 |
| ICP-SLAM                           | 0.01660  | 0.0204 |
| PointFusion (non-differentiable)   |  0.0071  | 0.0099 |
| PointFusion                        |  0.0072  | 0.0101 |
| KinectFusion (non-differentiable)  |   0.013  |  0.019 |
| KinectFusion                       |   0.016  |  0.021 |
| KinectFusion (non-differentiable)  |   0.013  |  0.019 |
| KinectFusion                       |   0.016  |  0.021 |
+------------------------------------+----------+--------+

## conclusion
a differentiable computational graph framwework that enables gradient-based learning for a lager set of localization and mapping based task, by providing explicit gradients with respect to the input image and dpeth maps. we demonstrate a diverse set of case studioes and showcase how the gradients propogate throughout the tracking, mapping, and fusion stages. future efforts will enable clifter_slam to be directly plugged into and optimized in conjuction with downstream task. clifter_slam can also enable a variety of sel-supervised learning applications, as any gradient-based learning architecture can now be equipped with a sense of __spatial understanding__


