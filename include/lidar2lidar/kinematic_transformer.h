#ifndef LIDAR2LIDAR_KINEMATIC_TRANSFORMER_H
#define LIDAR2LIDAR_KINEMATIC_TRANSFORMER_H

#include <mutex>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/Point.h>
#include "lidar2lidar/types.h"

namespace lidar2lidar {

class KinematicTransformer {
public:
    struct KinematicState {
        geometry_msgs::Point joint1;
        geometry_msgs::Point joint2;
        Eigen::Matrix4f transform;          // Final transform (boom_pos)
        Eigen::Matrix4f roll_transform;     // After roll
        Eigen::Matrix4f boom_yaw;            // After boom angle
        bool valid;
        
        KinematicState() : valid(false) {
            transform = Eigen::Matrix4f::Identity();
            roll_transform = Eigen::Matrix4f::Identity();
            boom_yaw = Eigen::Matrix4f::Identity();
        }
    };
    
    KinematicTransformer();
    ~KinematicTransformer() = default;
    
    // Update kinematic state from DH parameters
    void updateKinematics(const std::vector<double>& dh_params);
    
    // Transform point cloud based on kinematics
    CloudPtr transformCloud(
        const RawCloudPtr& input,
        double len_ratio,
        const Eigen::Vector3f& manual_offset,
        const Eigen::Vector3f& manual_rotation_rpy,
        const Eigen::Vector3f& additional_rotation_rpy
    );
    
    // Get current kinematic state (thread-safe)
    KinematicState getState() const;
    
    // Get midpoint for LiDAR mounting
    Eigen::Vector3f getMidpoint(double ratio, const Eigen::Vector3f& manual_offset) const;
    
    // Check if kinematics are ready
    bool isReady() const;
    
private:
    // DH transformation matrix
    Eigen::Matrix4f makeDHMatrix(float theta, float d, float a, float alpha) const;
    
    // Build orthonormal basis from direction vector
    Eigen::Matrix3f buildOrthonormalBasis(const Eigen::Vector3f& x_axis) const;
    
    // Convert RPY to rotation matrix
    Eigen::Matrix3f rpyToRotationMatrix(const Eigen::Vector3f& rpy) const;
    
    KinematicState current_state_;
    mutable std::mutex state_mutex_;
};

} // namespace lidar2lidar

#endif // LIDAR2LIDAR_KINEMATIC_TRANSFORMER_H
