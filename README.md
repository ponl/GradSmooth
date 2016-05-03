# GradSmooth

This point cloud smoothing algorithm evolves a point cloud so that the overall distance to measure (http://arxiv.org/pdf/1412.7197.pdf) is lowered. To accomplish this, we use the distance to measure function to induce a gradient flow on the point cloud. Using gradient descent, we evolve the point cloud. This is a multi-threaded implementation. Additionally, we use the nanoflann library (https://github.com/jlblancoc/nanoflann) for the nearest neighbor computations.

## Installation
GradSmooth uses CMake for building, therefore cmake is required. Additionally, GradSmooth has the following dependencies, however, all of these are included as submodules so no work is required on your part.
- gflags
- nanoflann
- easylogging
This makes it very easy to install GradSmooth. Simply do the following from the GradSmooth directory.

```
mkdir build
cd build
cmake ..
make
```

## Usage

GradSmooth uses the NumPy file format for point clouds. This makes it easy to run a point cloud through the smoother and then use the results in python. To run GradSmooth, use the following command

./gradsmooth path/to/input.npy path/to/output.npy

Additionally, GradSmooth has the following parameters which can be set.
    -iterations (Number of iterations to run the smoothing algorithm)
      type: int32 default: 10
    -max_leaf_size (Maximum number of points contained within a kd-tree leaf)
      type: int32 default: 10
    -num_neighbors (Number of nearest neighbors to use for knn-search)
      type: int32 default: 5
    -num_threads (Number of threads to use for the smoothing algorithm)
      type: int32 default: 1
    -step_size (Step size for gradient flow) type: double
      default: 0.1
      
The max_leaf_size and num_threads parameters are performance related while the other parameters determine how the smoothing will be performed. You will likely want to leave the max_leaf_size parameter at the default. If you are on a machine which can spare some threads, you may want to increase the num_threads parameter. 

The settings for the rest of the parameters will depend on the point cloud you are smoothing. Increasing the step_size will result in more aggressive and unstable smoothing. The number of interations will determine how many iterations the smoothing algorithm should be applied. Most of the smoothing will occur in the early iterations so setting this number very high may not result in much change from a lower value. In a future update, we will be adding convergence criteria so that we will automatically terminate the smoothing if the point cloud is no longer evolving a significant amount. The number of neighbors (num_neighbors) will also affect the stability of the smoothing. Setting a low value will result in smaller features of the point cloud (including noise) being accentuated. However, setting too large a value will result in many features being smoothed away.

## Contributing

Currently, this repository does not allow any outside contributors.

## License

GradSmooth is released under the GNU GENERAL PUBLIC LICENSE.
