#include "PointCloud.h"

/** Load a point cloud from a NumPy array file.
 *  param path The path to the input NumPy point cloud array.
 */
void PointCloud::load_cloud(std::string& path)
{
    LOG_DEBUG << "Loading cloud from: " << path;
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if(sizeof(Coordinate) != arr.word_size)
    {
        LOG_FATAL << "Mismatch in data sizes of NumPy file. Expected "
                   << sizeof(Coordinate) << " but got " << arr.word_size
                   << "(" << path << ")";
        return;
    }

    // Extract data
    dimension = arr.shape[1];
    unsigned count = arr.shape[0];
    LOG_INFO << "Found cloud with " << count
              << " points of dimension " << dimension;

    LOG_DEBUG << "Recasting data to Coordinate type";
    Coordinate* pdata = reinterpret_cast<Coordinate*>(arr.data);

    // Empty any contents of existing point cloud
    cloud.clear();

    // Load point cloud into vector
    LOG_DEBUG << "Filling point cloud vector with data from NumPy array";
    Point   point(dimension);
    for(unsigned i = 0; i < count; i++)
    {
        for(unsigned d = 0; d < dimension; d++)
        {
            point[d] = pdata[i * dimension + d];
        }
        cloud.push_back(point);
    }

    LOG_DEBUG << "Successfully loaded point cloud from file.";

    return;
}


/** Saves a point cloud to a NumPy array file.
 *  param path The path to save the point cloud as NumPy array.
 */
void PointCloud::save_cloud(std::string& path)
{
    unsigned size = cloud.size();
    const unsigned shape[] = {size, dimension};

    std::vector<Coordinate> flatdata(size * dimension);

    for(unsigned i = 0; i < size; i++)
    {
        for(unsigned d = 0; d < dimension; d++)
        {
            flatdata[i * dimension + d] = cloud[i][d];
        }
    }

    const Coordinate* pdata = flatdata.data();

    LOG_INFO << "Saving cloud with " << size << " points of dimension " << dimension
              << " to " << path;

    cnpy::npy_save<Coordinate>(path, pdata, &shape[0], 2);

    LOG_INFO << "Successfully saved cloud to path: " << path;
}

void PointCloud::copy_cloud(PointCloud& other)
{
    LOG_DEBUG << "Copying cloud data";
    Cloud* other_cloud = other.get_cloud();
    dimension = other.get_dimension();

    cloud.clear();

    LOG_DEBUG << "Filling in cloud";
    Point point(dimension);
    for(int i = 0; i < other_cloud -> size(); i++)
    {
        for(int d = 0; d < dimension; d++)
        {
            point[d] = (other_cloud -> at(i))[d];
        }
        cloud.push_back(point);
    }

}


void PointCloud::get_knn(Point p, const size_t k, Cloud& neighborhood, DistanceVector& distances)
{
    IndexVector indices(k);
    neighborhood = Cloud(k);
    distances = DistanceVector(k);
    p_kd_tree -> index -> knnSearch(&p[0], k, &indices[0], &distances[0]);

    for(unsigned i = 0; i < k; i++)
    {
        neighborhood[i] = cloud[indices[i]];
    }
}


void PointCloud::assign_kd_tree(KDTree* p, const size_t k, bool lock_neighbors, unsigned num_threads)
{
    p_kd_tree = p;
    if(lock_neighbors)
        build_neighborhood_map(k, num_threads);
}


void PointCloud::build_neighborhood_map(const size_t k, unsigned num_threads)
{
    LOG_INFO << "Building neighborhood map for locked neighbors";

    unsigned npoints = cloud.size();

    // Initialize Vectors
    neighbors = std::vector<Cloud>(num_threads);
    distances = std::vector<DistanceVector>(num_threads);

    neighbor_structure = NeighborVector(npoints);

    for(unsigned i = 0; i < npoints; i ++)
    {
        neighbor_structure[i] = std::make_pair(Cloud(k), DistanceVector(k));
    }

    for(unsigned i = 0; i < npoints; i += num_threads)
    {
        if((i % MAP_BUILDING_LOG_FREQUENCY) == 0)
            LOG_INFO << "Building neighborhood " << i << " of " << npoints;

        std::vector<std::thread> threads(num_threads);

        unsigned thread_count = std::min(num_threads, npoints - i);

        for(unsigned t = 0; t < thread_count; t++)
        {
            threads[t] = std::thread(construct_neighborhood_wrapper, this, i+t, k, t);
        }

        for(unsigned t= 0; t < thread_count; t++)
        {
            threads[t].join();
            NeighborPair neighbor_pair = std::make_pair(neighbors[t], distances[t]);
            neighbor_structure[i + t] = neighbor_pair;
        }
    }
}

void PointCloud::construct_neighborhood_wrapper(PointCloud* pcloud, unsigned indx, const size_t k, unsigned nthread)
{
    pcloud -> construct_neighborhood(indx, k, nthread);
}

void PointCloud::construct_neighborhood(unsigned indx, const size_t k, unsigned nthread)
{
    NeighborPair    neighbor_pair;

    get_knn(cloud[indx], k, neighbors[nthread], distances[nthread]);
}


void PointCloud::get_locked_knn(unsigned indx, Cloud& neighborhood, DistanceVector& distances)
{
    neighborhood = neighbor_structure[indx].first;
    distances = neighbor_structure[indx].second;
}
