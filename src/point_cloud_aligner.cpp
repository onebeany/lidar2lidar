#include "lidar2lidar/point_cloud_aligner.h"

// Include nano_gicp implementation headers
#include <nano_gicp/nano_gicp.hpp>

// OpenMP support
#ifdef _OPENMP
#include <omp.h>
#endif

// Include template implementations (header-only library)
#include <nano_gicp/impl/lsq_registration_impl.hpp>
#include <nano_gicp/impl/nano_gicp_impl.hpp>

namespace lidar2lidar {

PointCloudAligner::PointCloudAligner() 
    : icp_max_iterations_(50),
      icp_transformation_epsilon_(1e-8),
      icp_euclidean_fitness_epsilon_(1.0),
      gicp_max_iterations_(100),
      gicp_transformation_epsilon_(1e-4),
      gicp_correspondence_distance_(1.5) {
}

CloudPtr PointCloudAligner::cropWithStaticROI(
    const CloudPtr& input, const ROI& roi) const {
    
    CloudPtr output(new Cloud());
    
    if (!input || input->empty()) {
        ROS_WARN("PointCloudAligner: Empty input for static ROI cropping");
        return output;
    }
    
    pcl::CropBox<CloudPoint> crop_filter;
    crop_filter.setInputCloud(input);
    
    Eigen::Vector4f min_point(roi.min.x(), roi.min.y(), roi.min.z(), 1.0);
    Eigen::Vector4f max_point(roi.max.x(), roi.max.y(), roi.max.z(), 1.0);
    
    crop_filter.setMin(min_point);
    crop_filter.setMax(max_point);
    crop_filter.filter(*output);
    
    return output;
}

CloudPtr PointCloudAligner::cropWithDynamicROI(
    const CloudPtr& input,
    const Eigen::Vector3f& base_point,
    float x_range, float y_range, float z_range) const {
    
    CloudPtr output(new Cloud());
    
    if (!input || input->empty()) {
        ROS_WARN("PointCloudAligner: Empty input for dynamic ROI cropping");
        return output;
    }
    
    pcl::CropBox<CloudPoint> crop_filter;
    crop_filter.setInputCloud(input);
    
    Eigen::Vector4f min_point(
        base_point.x() - x_range / 2.0f,
        base_point.y() - y_range / 2.0f,
        base_point.z() - z_range / 2.0f,
        1.0
    );
    
    Eigen::Vector4f max_point(
        base_point.x() + x_range / 2.0f,
        base_point.y() + y_range / 2.0f,
        base_point.z() + z_range / 2.0f,
        1.0
    );
    
    crop_filter.setMin(min_point);
    crop_filter.setMax(max_point);
    crop_filter.filter(*output);
    
    return output;
}

CloudPtr PointCloudAligner::downsample(
    const CloudPtr& input,
    float leaf_x, float leaf_y, float leaf_z) const {
    
    CloudPtr output(new Cloud());
    
    if (!input || input->empty()) {
        ROS_WARN("PointCloudAligner: Empty input for downsampling");
        return output;
    }
    
    pcl::VoxelGrid<CloudPoint> voxel_filter;
    voxel_filter.setInputCloud(input);
    voxel_filter.setLeafSize(leaf_x, leaf_y, leaf_z);
    voxel_filter.filter(*output);
    
    return output;
}

AlignmentResult PointCloudAligner::alignICP(
    const CloudPtr& source,
    const CloudPtr& target,
    const Eigen::Matrix4f& initial_guess,
    int max_iterations,
    double transformation_epsilon,
    double euclidean_fitness_epsilon,
    double max_correspondence_distance) {
    
    AlignmentResult result;
    
    if (!source || source->empty() || !target || target->empty()) {
        ROS_WARN("PointCloudAligner: Empty source or target for ICP");
        result.success = false;
        return result;
    }
    
    try {
        pcl::IterativeClosestPoint<CloudPoint, CloudPoint> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaximumIterations(max_iterations);
        icp.setTransformationEpsilon(transformation_epsilon);
        icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);  // CRITICAL!
        
        CloudPtr aligned(new Cloud());
        icp.align(*aligned, initial_guess);
        
        result.aligned_cloud = aligned;
        result.final_transform = icp.getFinalTransformation();
        result.success = icp.hasConverged();
        result.fitness_score = icp.getFitnessScore();
        
        // Extract iterations count (if available)
        // Note: PCL ICP doesn't directly expose iteration count, 
        // so we use max_iterations as an approximation
        result.iterations = max_iterations;
        
        if (result.success) {
            ROS_DEBUG("ICP converged. Fitness score: %f", result.fitness_score);
        } else {
            ROS_WARN("ICP did not converge");
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("ICP alignment failed: %s", e.what());
        result.success = false;
    }
    
    return result;
}

AlignmentResult PointCloudAligner::alignGICP(
    const CloudPtr& source,
    const CloudPtr& target,
    const Eigen::Matrix4f& initial_guess,
    int max_iterations,
    double transformation_epsilon) {
    
    AlignmentResult result;
    
    if (!source || source->empty() || !target || target->empty()) {
        ROS_WARN("PointCloudAligner: Empty source or target for GICP");
        result.success = false;
        return result;
    }
    
    try {
        // Create nano_gicp instance
        nano_gicp::NanoGICP<CloudPoint, CloudPoint> gicp;
        
        // Configure GICP parameters
        gicp.setNumThreads(4);
        gicp.setCorrespondenceRandomness(20);
        gicp.setMaximumIterations(max_iterations);
        gicp.setTransformationEpsilon(transformation_epsilon);
        gicp.setMaxCorrespondenceDistance(gicp_correspondence_distance_);
        gicp.setRANSACIterations(50);
        
        // Set input clouds
        gicp.setInputSource(source);
        gicp.setInputTarget(target);
        
        // Perform alignment
        CloudPtr aligned(new Cloud());
        gicp.align(*aligned, initial_guess);
        
        // Extract results
        result.aligned_cloud = aligned;
        result.final_transform = gicp.getFinalTransformation();
        result.success = gicp.hasConverged();
        result.fitness_score = gicp.getFitnessScore();
        result.iterations = max_iterations;
        
        if (result.success) {
            ROS_DEBUG("GICP converged. Fitness score: %f", result.fitness_score);
        } else {
            ROS_WARN("GICP did not converge");
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("GICP alignment failed: %s", e.what());
        result.success = false;
    }
    
    return result;
}

void PointCloudAligner::setICPMaxIterations(int max_iter) {
    icp_max_iterations_ = max_iter;
}

void PointCloudAligner::setICPTransformationEpsilon(double epsilon) {
    icp_transformation_epsilon_ = epsilon;
}

void PointCloudAligner::setICPEuclideanFitnessEpsilon(double epsilon) {
    icp_euclidean_fitness_epsilon_ = epsilon;
}

void PointCloudAligner::setGICPMaxIterations(int max_iter) {
    gicp_max_iterations_ = max_iter;
}

void PointCloudAligner::setGICPTransformationEpsilon(double epsilon) {
    gicp_transformation_epsilon_ = epsilon;
}

void PointCloudAligner::setGICPCorrespondenceDistance(double distance) {
    gicp_correspondence_distance_ = distance;
}

} // namespace lidar2lidar
