#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#define ELPP_THREAD_SAFE

#include <iostream>
#include <string>
#include <vector>
#include <cnpy/cnpy.h>
#include <plog/Log.h>
#include "nanoflann/include/nanoflann.hpp"

#include "KDTreeAdaptor.h"

typedef double                                              Coordinate;
typedef std::vector<Coordinate>                             Point;
typedef std::vector<Point>                                  Cloud;
typedef KDTreeVectorOfVectorsAdaptor<Cloud, Coordinate>     KDTree;
typedef std::vector<size_t>                                 IndexVector;
typedef std::vector<Coordinate>                             DistanceVector;


class PointCloud
{
    private:
        Cloud                   cloud;
        unsigned                dimension;
        KDTree*                 p_kd_tree;


    public:
        void        load_cloud(std::string& path);
        void        save_cloud(std::string& path);
        void        get_point(unsigned i, Point& point) {point = cloud[i];}
        void        set_point(unsigned i, Point& point) {cloud[i] = point;}
        Cloud*      get_cloud()                         {return &cloud;}
        unsigned    get_dimension()                     {return dimension;}
        unsigned    get_size()                          {return cloud.size();}
        void        assign_kd_tree(KDTree* p)           {p_kd_tree = p;}
        void        get_knn(Point p, const size_t k, Cloud& neighborhood, DistanceVector& distances);
        void        copy_cloud(PointCloud& other);
};
#endif
