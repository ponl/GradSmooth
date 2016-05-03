#include "PointCloud.h"

/** Load a point cloud from a NumPy array file.
 *  param path The path to the input NumPy point cloud array.
 */
void PointCloud::load_cloud(std::string& path)
{
    LOG(INFO) << "Loading cloud from: " << path;
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if(sizeof(Coordinate) != arr.word_size)
    {
        LOG(FATAL) << "Mismatch in data sizes of NumPy file. Expected "
                   << sizeof(Coordinate) << " but got " << arr.word_size
                   << "(" << path << ")";
        return;
    }

    // Extract data
    dimension = arr.shape[1];
    unsigned count = arr.shape[0];
    LOG(INFO) << "Found cloud with " << count
              << " points of dimension " << dimension;

    LOG(DEBUG) << "Recasting data to Coordinate type";
    Coordinate* pdata = reinterpret_cast<Coordinate*>(arr.data);

    // Empty any contents of existing point cloud
    cloud.clear();

    // Load point cloud into vector
    LOG(DEBUG) << "Filling point cloud vector with data from NumPy array";
    Point   point(dimension);
    for(unsigned i = 0; i < count; i++)
    {
        for(unsigned d = 0; d < dimension; d++)
        {
            point[d] = pdata[i * dimension + d];
        }
        cloud.push_back(point);
    }

    LOG(INFO) << "Successfully loaded point cloud from file.";

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

    LOG(INFO) << "Saving cloud with " << size << " points of dimension " << dimension
              << " to " << path;

    cnpy::npy_save<Coordinate>(path, pdata, &shape[0], 2);

    LOG(INFO) << "Successfully saved cloud to path: " << path;
}

void PointCloud::copy_cloud(PointCloud& other)
{
    LOG(INFO) << "Copying cloud";
    LOG(DEBUG) << "Copying cloud information";
    Cloud* other_cloud = other.get_cloud();
    dimension = other.get_dimension();

    cloud.clear();

    LOG(DEBUG) << "Filling in cloud";
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


