// Copyright 2016 Patrick A. O'Neil
#ifndef INCLUDE_POINTCLOUD_H_
#define INCLUDE_POINTCLOUD_H_
#define ELPP_THREAD_SAFE


#include <cnpy/cnpy.h>
#include <plog/Log.h>
#include <KDTreeAdaptor.h>

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <thread>
#include <utility>

#include <nanoflann/include/nanoflann.hpp>

typedef double                                              Coordinate;     /**<Underlying data type for coordinates*/
typedef std::vector<Coordinate>                             Point;          /**<Multidimensional point*/
typedef std::vector<Point>                                  Cloud;          /**<Vector of points which represents a cloud*/
typedef KDTreeVectorOfVectorsAdaptor<Cloud, Coordinate>     KDTree;         /**<k-d Tree adaptor for querying neighbors*/
typedef std::vector<size_t>                                 IndexVector;    /**<Vector to store indices*/
typedef std::vector<Coordinate>                             DistanceVector; /**<Vector of Euclidean distances*/
typedef std::vector<Cloud>                                  NeighborVector; /**<Vector to store neighbors*/

const unsigned MAP_BUILDING_LOG_FREQUENCY = 10000;

/**
 * Stores the point cloud, provides neighborhood search, and supports numpy loading and saving. This is the class which stores
 * all of the points in the cloud. It supports arbitrary dimensional point clouds.
 */

class PointCloud
{
    private:
        Cloud                           cloud;              /**<Stores all the points in the cloud.*/
        unsigned                        dimension;          /**<Dimension of the stored point cloud.*/
        KDTree*                         p_kd_tree;          /**<Pointer to the constructed k-d tree.*/
        NeighborVector                  neighbor_structure; /**<Persistent storage for neighbors. Used with neighbor lock.*/
        NeighborVector                  neighbors;          /**<Multi-threaded temporary storage of neighbors.*/
        std::vector<DistanceVector>     distances;          /**<Multi-threaded temporary storage of distances.*/
    public:
        /**
         * Load the point cloud from a NumPy file.
         * @param path The path to the NumPy file storing the point cloud.
         */
        void        load_cloud(std::string& path);

        /**
         * Saves the point cloud to a NumPy file.
         * @param path The path to save the NumPy file.
         */
        void        save_cloud(std::string& path);

        /**
         * Gets the point at index i and stores in point.
         * @param i The index of the point.
         * @param point The Point object to store the desired point.
         */
        void        get_point(unsigned i, Point& point) {point = cloud[i];}

        /**
         * Overwrites the point at the given index with the supplied point.
         * @param i Index of the point to be overwritten.
         * @param point The new value for the point.
         */
        void        set_point(unsigned i, Point& point) {cloud[i] = point;}

        /**
         * Returns a pointer to the underlying cloud data.
         * @return Pointer to the underlying cloud vector data.
         */
        Cloud*      get_cloud()                         {return &cloud;}

        /**
         * Returns the dimension of the point cloud.
         * @return The dimension of the point cloud.
         */
        unsigned    get_dimension()                     {return dimension;}

        /**
         * Returns the size of the point cloud.
         * @return The number of points in the point cloud.
         */
        unsigned    get_size()                          {return cloud.size();}

        /**
         * Gives the point cloud a pointer to the k-d tree used for querying neighbors.
         * @param p Pointer to the k-d tree.
         * @param k Number of neighbors used in the k-d tree.
         * @param lock_neighbors Determines whether neighbors will update during the flow or stay locked throughout.
         * @param num_threads The number of threads to be used while running the flow.
         */
        void        assign_kd_tree(KDTree* p, const size_t k, bool lock_neighbors, unsigned num_threads);

        /**
         * Gets the k-nearest neighbors to a query point.
         * @param p The point to query for the k-nearest neighbors.
         * @param k The number of neighbors for which to search.
         * @param neighborhood The data structure in which to store the neighbors.
         * @param distances The data structure to store the distances to all the neighbors.
         */
        void        get_knn(Point& p, const size_t k, Cloud& neighborhood, DistanceVector& distances);

        /**
         * Gets the k-nearest neighbors to a query point using locked neighbors.
         * @param p The point to query for the k-nearest neighbors.
         * @param indx The index of the query point.
         * @param neighborhood The data structure in which to store the neighbors.
         * @param distances The data structure to store the distances to all the neighbors.
         */
        void        get_locked_knn(Point& p, unsigned indx, Cloud& neighborhood, DistanceVector& distances);

        /**
         * Copies the point cloud data to another PointCloud object.
         * @param other The PointCloud to which this cloud's data will be copied.
         */
        void        copy_cloud(PointCloud& other);

        /**
         * Builds the persistent neighborhood map when neighbors are locked.'
         * @param k Number of neighbors to use.
         * @param num_threads Number of threads to use when building the map.
         */
        void        build_neighborhood_map(const size_t k, unsigned num_threads);

        /**
         * Finds the k-nearest neighbors of a point and stores them in the neighbors structure.
         * @param indx Index of the point to query against.
         * @param k The number of neighbors for which to search.
         * @param nthread The thread number for which this call is being made.
         */
        void        construct_neighborhood(unsigned indx, const size_t k, unsigned nthread);

        /**
         * Wrapper for multithreaded computation of the neighborhood map.
         * @param pcloud Pointer to the point cloud to use for neighborhood searches.
         * @param indx Index of the point to be queried against.
         * @param k The number of neighbors for which to search.
         * @param nthread The thread number for which this call is being made.
         */
        static void construct_neighborhood_wrapper(PointCloud* pcloud, unsigned indx, const size_t k, unsigned nthread);
};

#endif  // INCLUDE_POINTCLOUD_H_
