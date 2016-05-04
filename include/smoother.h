#ifndef SMOOTHER_H
#define SMOOTHER_H

#define ELPP_THREAD_SAFE

#include <iostream>
#include <math.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>

#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Eigenvalues>
#include "easyloggingpp/src/easylogging++.h"


#include "PointCloud.h"

typedef std::vector<Point>      VectorList;

class Smoother
{
    private:
        bool                            normal_projection;
        size_t                          num_neighbors;
        unsigned                        nthreads;
        unsigned                        dimension;
        unsigned                        codimension;
        double                          step_size;
        std::vector<Point>              updated_points;
        std::vector<Point>              gradients;
        std::vector<Cloud>              neighborhoods;
        std::vector<DistanceVector>     distances;

        void get_gradient(Point& point, Cloud& neighborhood, Point& gradient);
        void update_point(Point& point, Point& gradient);
        void get_weighted_barycenter(Point& query_point, Cloud& neighborhood, Point& barycenter, const double sigma);
        void get_frame(Point& query_point, Cloud& neighborhood, const double sigma, VectorList& normals);
        void flow_point(unsigned thread_id, PointCloud& cloud);

        // Static Functions
        static Coordinate get_squared_distance(Point& p0, Point& p1);
        static void flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud);

    public:
        Smoother(size_t num_neighbors_, unsigned dimension_, unsigned codimension_, unsigned nthreads_, double step_size_, bool normal_projection_);
        void smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T);
};

#endif
