#ifndef LIDAR2LIDAR_TYPES_H
#define LIDAR2LIDAR_TYPES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lidar2lidar {

// Custom point type for raw sensor data (based on MLX)
namespace point_types {
    struct EIGEN_ALIGN16 RawPoint {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t offset_time;
        uint16_t ring;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace point_types

} // namespace lidar2lidar

// Register the point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(lidar2lidar::point_types::RawPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint32_t, offset_time, offset_time)
    (uint16_t, ring, ring)
)

namespace lidar2lidar {

// Type aliases for convenience
using RawPoint = point_types::RawPoint;
using RawCloud = pcl::PointCloud<RawPoint>;
using RawCloudPtr = typename RawCloud::Ptr;
using RawCloudConstPtr = typename RawCloud::ConstPtr;

using CloudPoint = pcl::PointXYZINormal;
using Cloud = pcl::PointCloud<CloudPoint>;
using CloudPtr = typename Cloud::Ptr;
using CloudConstPtr = typename Cloud::ConstPtr;

// Alignment metrics structure
struct AlignMetrics {
    size_t source_points = 0;
    size_t source_coarse_points = 0;
    size_t source_fine_points = 0;
    size_t target_points = 0;
    size_t target_roi_points = 0;
    size_t target_coarse_points = 0;
    size_t target_fine_points = 0;
    bool coarse_icp_converged = false;
    bool fine_icp_converged = false;
    bool gicp_converged = false;
    double coarse_icp_fitness = -1.0;
    double fine_icp_fitness = -1.0;
    double gicp_fitness = -1.0;
    double translation_norm = 0.0;
    double rotation_angle_deg = 0.0;
    double pre_icp_time_us = 0.0;
    double icp_processing_time_us = 0.0;
};

// ROI (Region of Interest) structure
struct ROI {
    Eigen::Vector3f min;
    Eigen::Vector3f max;
    
    ROI() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
    ROI(const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt) 
        : min(min_pt), max(max_pt) {}
    
    bool isValid() const {
        return (min.x() <= max.x()) && (min.y() <= max.y()) && (min.z() <= max.z());
    }
};

} // namespace lidar2lidar

#endif // LIDAR2LIDAR_TYPES_H
