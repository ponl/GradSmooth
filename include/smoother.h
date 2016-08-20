#ifndef SMOOTHER_H
#define SMOOTHER_H

#define ELPP_THREAD_SAFE

#include <iostream>
#include <math.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>

#include <plog/Log.h>
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Eigenvalues>

#include "PointCloud.h"

typedef std::vector<Point>      VectorList;

/**
 * k-NN gradient smoothing algorithm. This class provides the ability to smooth
 */

class Smoother
{
    private:
        bool                            normal_projection;
        bool                            lock_neighbors;
        size_t                          num_neighbors;
        unsigned                        nthreads;
        unsigned                        dimension;
        unsigned                        codimension;
        double                          step_size;
        std::vector<Point>              updated_points;
        std::vector<unsigned>           point_indices;
        std::vector<Point>              gradients;
        std::vector<Cloud>              neighborhoods;
        std::vector<DistanceVector>     distances;

        void get_gradient(Point& point, Cloud& neighborhood, Point& gradient);
        void update_point(Point& point, Point& gradient);
        void get_weighted_barycenter(Point& query_point, Cloud& neighborhood, DistanceVector& distances, Point& barycenter, const double sigma);
        void get_frame(Point& query_point, Cloud& neighborhood, DistanceVector& distances, const double sigma, VectorList& normals);
        void flow_point(unsigned thread_id, PointCloud& cloud);

        // Static Functions
        static Coordinate get_squared_distance(Point& p0, Point& p1);
        static void flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud);

    public:
        Smoother(size_t num_neighbors_, unsigned dimension_, unsigned codimension_, unsigned nthreads_, double step_size_, bool normal_projection_, bool lock_neighbors_);
        void smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T);
};

#endif
