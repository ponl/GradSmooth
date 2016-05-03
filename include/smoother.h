#ifndef SMOOTHER_H
#define SMOOTHER_H

#include "easylogging++.h"
#include "PointCloud.h"

class Smoother
{
    private:
        void get_gradient(Point& point, Cloud& neighborhood, Point& gradient);
        void update_point(Point& point, Point& gradient, const double step_size);

    public:
        void smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const size_t k, const unsigned T, const double step_size);
};

#endif
