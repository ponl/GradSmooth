#include "smoother.h"

Smoother::Smoother(size_t num_neighbors_, unsigned dimension_, unsigned codimension_, unsigned nthreads_, double step_size_, bool normal_projection_, bool lock_neighbors_)
{
    num_neighbors = num_neighbors_;
    dimension = dimension_;
    codimension = codimension_;
    nthreads = nthreads_;
    step_size = step_size_;
    normal_projection = normal_projection_;
    lock_neighbors = lock_neighbors_;

    // Initialize vectors for threading
    updated_points = std::vector<Point>(nthreads, Point(dimension));
    point_indices = std::vector<unsigned>(nthreads);
    gradients = std::vector<Point>(nthreads, Point(dimension));
    neighborhoods = std::vector<Cloud>(nthreads, Cloud(num_neighbors));
    distances = std::vector<DistanceVector>(nthreads, DistanceVector(num_neighbors));
}


void Smoother::flow_point(unsigned thread_id, PointCloud& cloud)
{
    // Find nearest neighbors in original point cloud
    LOG_DEBUG << "Getting neighborhood from k-d tree";
    if(lock_neighbors)
    {
        cloud.get_locked_knn(point_indices[thread_id], neighborhoods[thread_id], distances[thread_id]);
    }
    else
    {
        cloud.get_knn(updated_points[thread_id], num_neighbors, neighborhoods[thread_id], distances[thread_id]);
    }

    // Compute the gradient
    LOG_DEBUG << "Computing the gradient";
    get_gradient(updated_points[thread_id], neighborhoods[thread_id], gradients[thread_id]);

    if(normal_projection)
    {
        VectorList normal_vectors(codimension);

        // TODO: Abstract out sigma allow to be set
        LOG_DEBUG << "Getting frame for query point";
        get_frame(updated_points[thread_id], neighborhoods[thread_id], distances[thread_id], 0.1, normal_vectors);

        // Compute dot products and vector norms
        LOG_DEBUG << "Projecting gradient onto normals";
        std::vector<Coordinate> dots(codimension);
        std::vector<Coordinate> norms(codimension);
        for(unsigned i = 0; i < codimension; i++)
        {
            Coordinate dproduct = 0.;
            Coordinate norm = 0.;
            for(unsigned d = 0; d < dimension; d++)
            {
                dproduct += gradients[thread_id][d] * normal_vectors[i][d];
                norm += pow(normal_vectors[i][d], 2);
            }
            dots[i] = dproduct;
            norms[i] = norm;
        }

        // Project gradient onto normals
        Point ngradient(dimension);
        for(unsigned i = 0; i < codimension; i++)
        {
            for(unsigned d = 0; d < dimension; d++)
            {
                ngradient[d] += normal_vectors[i][d] * dots[i] / norms[i];
            }
        }
        LOG_DEBUG << "Setting new gradient.";
        gradients[thread_id] = ngradient;
    }

    // Move the point along the gradient
    LOG_DEBUG << "Updating point in cloud";
    update_point(updated_points[thread_id], gradients[thread_id]);
}


void Smoother::flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud)
{
    smoother -> flow_point(thread_id, cloud);
}


void Smoother::smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T)
{
    LOG_INFO << "Beginning smoothing operation";
    LOG_INFO << "Number Neighbors: " << num_neighbors;
    LOG_INFO << "Step Size: " << step_size;
    LOG_INFO << "Normal Projection: " << (normal_projection ? "True": "False");
    LOG_INFO << "Iterations: " << T;

    unsigned num_points = cloud.get_size();
    unsigned dimension  = cloud.get_dimension();

    Smoother* smoother = this;
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < T; t++)
    {
        LOG_INFO << "Smoothing step " << t << " of " << T;
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
                point_indices[n] = point_index;

                // Add thread
                threads.push_back(std::thread(&Smoother::flow_point_wrapper, smoother, n, std::ref(cloud)));
            }

            LOG_DEBUG << "Processing points " << i << " to " << i + nthr_round << " of "<< num_points;

            // Synchronize all threads
            for(auto& th: threads) th.join();

            // Assign all updated points
            for(unsigned n = 0; n < nthr_round; n++)
            {
                evolved.set_point(i + n, updated_points[n]);
            }
        }
    }

    LOG_INFO << "Finished smoothing operation";
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

    // Normalize by number of neighbors
    for(unsigned d = 0; d < dim; d++)
    {
        gradient[d] /= (k / 2.0);
    }
}


// Find the Euclidean distance between two points
Coordinate Smoother::get_squared_distance(Point& p0, Point& p1)
{
    unsigned dim = p0.size();
    if(dim != p1.size())
    {
        LOG_FATAL << "Encountered two points with different dimensions!";
    }
    Coordinate distance = 0.;
    for(unsigned d = 0; d < dim; d++)
    {
        distance += pow(p0[d] - p1[d], 2);
    }
    return distance;
}


// Find the weighted barycenter of a neighborhood
void Smoother::get_weighted_barycenter(Point& query_point, Cloud& neighborhood, DistanceVector& distances, Point& barycenter, const double sigma)
{
    DistanceVector weights(num_neighbors);

    // Compute normalization factor
    Coordinate normalizer = (*std::max_element(distances.begin(), distances.end())) * pow(sigma, 2);

    if(normalizer == 0)
    {
        LOG_WARNING << "Encountered point with max distance from neighbors 0!";
        barycenter = query_point;
        return;
    }

    // Compute weights
    for(unsigned i = 0; i < num_neighbors; i++)
    {
        weights[i] = exp(-1 * distances[i]*distances[i] / normalizer);
    }

    // Normalize weights
    Coordinate total_weight = std::accumulate(weights.begin(), weights.end(), 0.);

    if(total_weight == 0)
    {
        LOG_WARNING << "Encountered point whose neighbor weights were all zero!";
        barycenter = query_point;
        return;
    }

    // Initialize the barycenter
    for(unsigned d = 0; d < dimension; d++)
    {
        barycenter[d] = 0;
    }

    // Compute barycenter as weighted sum of points
    for(unsigned i = 0; i < num_neighbors; i++)
    {
        weights[i] /= total_weight;
        for(unsigned d = 0; d < dimension; d++)
        {
            barycenter[d] += neighborhood[i][d] * weights[i];
        }
    }
}


// Computes the local frame around a point
// TODO: Switch to using current position of neighbors
void Smoother::get_frame(Point& query_point, Cloud& neighborhood, DistanceVector& distances, const double sigma, VectorList& normals)
{
    LOG_DEBUG << "Computing coordinate frame.";
    Point barycenter(dimension);
    LOG_DEBUG << "Getting weighted barycenter";
    get_weighted_barycenter(query_point, neighborhood, distances, barycenter, sigma);

    LOG_DEBUG << "Computing neighbor to barycenter matrix";
    Eigen::MatrixXd    b2n(num_neighbors, dimension);

    // Get matrix of vectors from neighbors to barycenter
    for(unsigned i = 0; i < num_neighbors; i++)
    {
        for(unsigned d = 0; d < dimension; d++)
        {
            b2n(i,d) = neighborhood[i][d] - barycenter[d];
        }
    }

    LOG_DEBUG << "Centering neighbor to barycenter matrix";
    Eigen::MatrixXd centered = b2n.rowwise() - b2n.colwise().mean();

    LOG_DEBUG << "Computing covariance matrix";
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(b2n.rows() - 1);

    Eigen::EigenSolver<Eigen::MatrixXd> es(cov);

    const Eigen::VectorXcd& evalues  = es.eigenvalues();
    const Eigen::MatrixXcd&  evectors = es.eigenvectors();

    // Determine order of eigenvalues
    std::vector<size_t> evalue_indices(evalues.size());

    for(size_t i = 0; i < evalues.size(); i++) evalue_indices[i] = i;

    std::sort(evalue_indices.begin(), evalue_indices.end(), [&evalues](size_t i1, size_t i2 ){return evalues[i1].real() < evalues[i2].real();});

    // Assign normal and tangent vectors
    Point new_vector(dimension);
    bool normal_vector;
    for(unsigned i = 0; i < codimension; i++)
    {
        size_t idx = evalue_indices[i];
        Eigen::VectorXcd  ev = evectors.col(idx);
        for(unsigned d = 0; d < dimension; d++)
        {
            new_vector[d] = ev(d).real();
        }
        normals[i] = new_vector;
    }
}



