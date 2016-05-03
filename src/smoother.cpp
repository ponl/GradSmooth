#include "smoother.h"

void Smoother::smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const size_t k, const unsigned T, const double step_size)
{
    LOG(INFO) << "Beginning smoothing operation";

    unsigned num_points = cloud.get_size();
    unsigned dimension  = cloud.get_dimension();

    Point point(dimension);
    Point gradient(dimension);
    Cloud neighborhood(k);
    DistanceVector distances(k);
    for(unsigned t = 0; t < T; t++)
    {
        LOG(INFO) << "Smoothing step " << t << " of " << T;
        for(unsigned i = 0; i < num_points; i++)
        {
            LOG(DEBUG) << "Processing point " << i << " of " << num_points;

            // Get Point from evolved cloud
            evolved.get_point(i, point);

            // Find nearest neighbors in original point cloud
            cloud.get_knn(point, k, neighborhood, distances);

            // Compute the gradient
            get_gradient(point, neighborhood, gradient);

            // Move the point along the gradient
            update_point(point, gradient, step_size);

            // Set the point in the evolved point cloud
            evolved.set_point(i, point);
        }
    }

    LOG(INFO) << "Finished smoothing operation";
}


// Updates point in the direction of the gradient
void Smoother::update_point(Point& point, Point& gradient, const double step_size)
{
    for(unsigned d = 0; d < point.size(); d++)
    {
        point[d] -= step_size * gradient[d];
    }
}


// Determines the gradient from a neighborhood
void Smoother::get_gradient(Point& point, Cloud& neighborhood, Point& gradient)
{
    unsigned dim = gradient.size();
    float k = (float)neighborhood.size();

    for(unsigned d = 0; d < dim; d++)
        gradient[d] = 0;

    for(unsigned i = 0; i < neighborhood.size(); i++)
    {
        for(unsigned d = 0; d < dim; d++)
        {
            gradient[d] += (point[d] - neighborhood[i][d]);
        }
    }

    for(unsigned d = 0; d < dim; d++)
    {
        gradient[d] /= (k / 2.0);
    }
}
