#ifndef SMOOTHER_H
#define SMOOTHER_H

#define ELPP_THREAD_SAFE

#include <iostream>
#include <math.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <numeric>

#include <plog/Log.h>
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Eigenvalues>

#include "PointCloud.h"

typedef std::vector<Point>      VectorList;

/**
 * k-NN gradient smoothing algorithm. This class provides the ability to smooth
 */

class Smoother
{
    private:
        bool                            normal_projection; /**<Determines whether to project the gradient along the normal.*/
        bool                            lock_neighbors;    /**<Determines whether to update the neighbors during the flow or keep them locked.*/
        size_t                          num_neighbors;     /**<Number of neighbors to use for neighborhood construction.*/
        unsigned                        nthreads;          /**<Number of threads to use for the flow.*/
        unsigned                        dimension;         /**<Dimension of the underlying manifold from which the data was sampled.*/
        unsigned                        codimension;       /**<Codimension of the underlying manifold.*/
        double                          step_size;         /**<Step size used for the gradient flow.*/
        std::vector<Point>              updated_points;    /**<Points which have been updated in this batch.*/
        std::vector<unsigned>           point_indices;     /**<Indices of points being updated in this batch.*/
        std::vector<Point>              gradients;         /**<Gradients of points in this batch.*/
        std::vector<Cloud>              neighborhoods;     /**<Neighborhoods of points in this batch.*/
        std::vector<DistanceVector>     distances;         /**<Distances to neighbors of the points in this batch.*/

        /**
         * Determine the gradient for the query point.
         * @param point The point for which the gradient will be computed.
         * @param neighborhood The neighborhood of the query point.
         * @param gradient The Point where the computed gradient will be stored.
         */
        void get_gradient(Point& point, Cloud& neighborhood, Point& gradient);

        /**
         * Moves the point in the direction of the supplied gradient.
         * @param point The point which will be moved.
         * @param gradient The direction to move the point.
         */
        void update_point(Point& point, Point& gradient);

        /**
         * Computes the weighted barycenter of the query point.
         * @param query_point The point for which the weighted barycenter will be computed.
         * @param neighborhood The k-neighbors of the query point.
         * @param distances The distances to the k-neighbors of the query point.
         * @param barycenter The Point in which the weighted barycenter will be stored.
         * @param sigma Weighting factor for the barycenter. Higher values indicate faster fall-off for distant points.
         */
        void get_weighted_barycenter(Point& query_point, Cloud& neighborhood, DistanceVector& distances, Point& barycenter, const double sigma);

        /**
         * Gets the estimated local coordinate frame of the query point.
         * @param query_point The point for which the frame will be computed.
         * @param neighborhood The k-nearest neighbors of the query point.
         * @param distances The distances to the k-nearest neighbors.
         * @param sigma Weighting factor for the barycenter. Higher values indicate faster fall-off for distant points.
         * @param normals The normal vectors of the computed local frame.
         */
        void get_frame(Point& query_point, Cloud& neighborhood, DistanceVector& distances, const double sigma, VectorList& normals);

        /**
         * Updates the specified point of the batch of points currently being updated.
         * @param thread_id ID of the thread making this call.
         * @param cloud The point cloud undergoing smoothing.
         */
        void flow_point(unsigned thread_id, PointCloud& cloud);

        /**
         * Wrapper for multi-threaded point flowing.
         * @param smoother Pointer to the gradient flow smoother performing the smoothing operation.
         * @param thread_id ID of the thread making this call.
         * @param cloud The point cloud undergoing smoothing.
         */
        static void flow_point_wrapper(Smoother* smoother, unsigned thread_id, PointCloud& cloud);

        /**
         * Computes the squared distance between two points.
         * @param p0 One of the points for which the distance will be computed.
         * @param p1 One of the points for which the distance will be computed.
         */
        static Coordinate get_squared_distance(Point& p0, Point& p1);

    public:
        /**
         * Constructor of the Smoother. Will initialize all the objects in preparation for a multi-threaded flow.
         * @param num_neighbors_ Number of neighbors to use in the flow.
         * @param dimension_ Dimension of the underlying manifold from which the point cloud was sampled.
         * @param codimension_ Codimension of the underlying manifold from which the point cloud was sampled.
         * @param nthreads_ Number of threads to use when running the flow.
         * @param step_size_ Step size to use for the gradient flow.
         * @param normal_projection_ Determines whether the gradient will be projected onto approximated normals before flowing.
         * @param lock_neighbors_ Determines whether neighbors will be updated during the flow or locked.
         */
        Smoother(size_t num_neighbors_, unsigned dimension_, unsigned codimension_, unsigned nthreads_, double step_size_, bool normal_projection_, bool lock_neighbors_);

        /**
         * Perform smoothing on a point cloud.
         * @param cloud PointCloud for which smoothing will be applied.
         * @param evolved PointCloud for which the smoothed version of cloud will be stored.
         * @param T Number of iterations to run the smoothing.
         */
        void smooth_point_cloud(PointCloud& cloud, PointCloud& evolved, const unsigned T);
};

#endif
