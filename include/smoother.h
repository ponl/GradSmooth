#ifndef SMOOTHER_H
#define SMOOTHER_H

#include <vector>
#include <thread>
#include <algorithm>

#include "easylogging++.h"
#include "PointCloud.h"

class Smoother
{
    private:
        size_t                          num_neighbors;
        unsigned                        nthreads;
        unsigned                        dimension;
        double                          step_size;
        std::vector<Point>              updated_points;
        std::vector<Point>              gradients;
        std::vector<Cloud>              neighborhoods;
        std::vector<DistanceVector>     distances;

        void get_gradient(Point& point, Cloud& neighborhood, Point& gradient);
        void update_point(Point& point, Point& gradient);
        void flow_point(unsigned thread_id, PointCloud& cloud);
        static void flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud);

    public:
        Smoother(size_t num_neighbors_, unsigned dimension_, unsigned nthreads_, double step_size_);
        void smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T);
};

#endif
