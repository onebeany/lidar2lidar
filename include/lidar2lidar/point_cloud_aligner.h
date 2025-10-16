#ifndef LIDAR2LIDAR_POINT_CLOUD_ALIGNER_H
#define LIDAR2LIDAR_POINT_CLOUD_ALIGNER_H

#include "lidar2lidar/types.h"
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <ros/console.h>

// Forward declaration for nano_gicp
namespace nano_gicp {
    template<typename PointSource, typename PointTarget>
    class NanoGICP;
}

namespace lidar2lidar {

// Alignment result structure
struct AlignmentResult {
    CloudPtr aligned_cloud;
    Eigen::Matrix4f final_transform;
    bool success;
    double fitness_score;
    int iterations;
    
    // Additional metrics for detailed reporting
    size_t source_points;
    size_t target_points;
    
    AlignmentResult() 
        : success(false), 
          fitness_score(std::numeric_limits<double>::max()),
          iterations(0),
          source_points(0),
          target_points(0) {
        aligned_cloud.reset(new Cloud());
        final_transform = Eigen::Matrix4f::Identity();
    }
};

class PointCloudAligner {
public:
    PointCloudAligner();
    
    // ROI cropping
    CloudPtr cropWithStaticROI(const CloudPtr& input, const ROI& roi) const;
    CloudPtr cropWithDynamicROI(
        const CloudPtr& input, 
        const Eigen::Vector3f& base_point,
        float x_range, float y_range, float z_range) const;
    
    // Downsampling
    CloudPtr downsample(
        const CloudPtr& input,
        float leaf_x, float leaf_y, float leaf_z) const;
    
    // ICP alignment
    AlignmentResult alignICP(
        const CloudPtr& source,
        const CloudPtr& target,
        const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
        int max_iterations = 50,
        double transformation_epsilon = 1e-8,
        double euclidean_fitness_epsilon = 1.0,
        double max_correspondence_distance = 0.5);
    
    // GICP alignment
    AlignmentResult alignGICP(
        const CloudPtr& source,
        const CloudPtr& target,
        const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
        int max_iterations = 100,
        double transformation_epsilon = 1e-4);
    
    // Set alignment parameters
    void setICPMaxIterations(int max_iter);
    void setICPTransformationEpsilon(double epsilon);
    void setICPEuclideanFitnessEpsilon(double epsilon);
    
    void setGICPMaxIterations(int max_iter);
    void setGICPTransformationEpsilon(double epsilon);
    void setGICPCorrespondenceDistance(double distance);
    
private:
    // ICP parameters
    int icp_max_iterations_;
    double icp_transformation_epsilon_;
    double icp_euclidean_fitness_epsilon_;
    
    // GICP parameters
    int gicp_max_iterations_;
    double gicp_transformation_epsilon_;
    double gicp_correspondence_distance_;
};

} // namespace lidar2lidar

#endif // LIDAR2LIDAR_POINT_CLOUD_ALIGNER_H
