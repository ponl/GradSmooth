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

typedef double                                              Coordinate;
typedef std::vector<Coordinate>                             Point;
typedef std::vector<Point>                                  Cloud;
typedef KDTreeVectorOfVectorsAdaptor<Cloud, Coordinate>     KDTree;
typedef std::vector<size_t>                                 IndexVector;
typedef std::vector<Coordinate>                             DistanceVector;
typedef std::vector<Cloud>                                  NeighborVector;

const unsigned MAP_BUILDING_LOG_FREQUENCY = 10000;

class PointCloud
{
    private:
        Cloud                           cloud;
        unsigned                        dimension;
        KDTree*                         p_kd_tree;
        NeighborVector                  neighbor_structure;
        std::vector<Cloud>              neighbors;
        std::vector<DistanceVector>     distances;
    public:
        void        load_cloud(std::string& path);
        void        save_cloud(std::string& path);
        void        get_point(unsigned i, Point& point) {point = cloud[i];}
        void        set_point(unsigned i, Point& point) {cloud[i] = point;}
        Cloud*      get_cloud()                         {return &cloud;}
        unsigned    get_dimension()                     {return dimension;}
        unsigned    get_size()                          {return cloud.size();}
        void        assign_kd_tree(KDTree* p, const size_t k, bool lock_neighbors, unsigned num_threads);
        void        get_knn(Point& p, const size_t k, Cloud& neighborhood, DistanceVector& distances);
        void        get_locked_knn(Point& p, unsigned indx, Cloud& neighborhood, DistanceVector& distances);
        void        copy_cloud(PointCloud& other);
        void        build_neighborhood_map(const size_t k, unsigned num_threads);
        void        construct_neighborhood(unsigned indx, const size_t k, unsigned nthread);
        static void construct_neighborhood_wrapper(PointCloud* pcloud, unsigned indx, const size_t k, unsigned nthread);
};

#endif  // INCLUDE_POINTCLOUD_H_
