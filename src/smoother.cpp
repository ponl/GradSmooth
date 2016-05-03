#include "smoother.h"

Smoother::Smoother(size_t num_neighbors_, unsigned dimension_, unsigned nthreads_, double step_size_)
{
    num_neighbors = num_neighbors_;
    dimension = dimension_;
    nthreads = nthreads_;
    step_size = step_size_;

    // Initialize vectors for threading
    updated_points = std::vector<Point>(nthreads, Point(dimension));
    gradients = std::vector<Point>(nthreads, Point(dimension));
    neighborhoods = std::vector<Cloud>(nthreads, Cloud(num_neighbors));
    distances = std::vector<DistanceVector>(nthreads, DistanceVector(num_neighbors));
}


void Smoother::flow_point(unsigned thread_id, PointCloud& cloud)
{
    // Find nearest neighbors in original point cloud
    cloud.get_knn(updated_points[thread_id], num_neighbors, neighborhoods[thread_id], distances[thread_id]);

    // Compute the gradient
    get_gradient(updated_points[thread_id], neighborhoods[thread_id], gradients[thread_id]);

    // Move the point along the gradient
    update_point(updated_points[thread_id], gradients[thread_id]);
}


void Smoother::flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud)
{
    smoother -> flow_point(thread_id, cloud);
}


void Smoother::smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T)
{
    LOG(INFO) << "Beginning smoothing operation";

    unsigned num_points = cloud.get_size();
    unsigned dimension  = cloud.get_dimension();

    Smoother* smoother = this;
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < T; t++)
    {
        LOG(INFO) << "Smoothing step " << t << " of " << T;
        for(unsigned i = 0; i < num_points; i += nthreads)
        {
            threads.clear();

            unsigned nthr_round = std::min(num_points - i, nthreads);
            for(unsigned n = 0; n < nthr_round; n++)
            {
                unsigned point_index = i + n;
                Point current_point;
                evolved.get_point(point_index, current_point);
                updated_points[n] = current_point;

                // Add thread
                threads.push_back(std::thread(&Smoother::flow_point_wrapper, smoother, n, std::ref(cloud)));
            }

            LOG(DEBUG) << "Processing points " << i << " to " << i + nthr_round << " of "<< num_points;

            // Synchronize all threads
            for(auto& th: threads) th.join();

            // Assign all updated points
            for(unsigned n = 0; n < nthr_round; n++)
            {
                evolved.set_point(i + n, updated_points[n]);
            }


            /*
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
            */
        }
    }

    LOG(INFO) << "Finished smoothing operation";
}


// Updates point in the direction of the gradient
void Smoother::update_point(Point& point, Point& gradient)
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
