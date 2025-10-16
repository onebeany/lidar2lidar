#include "lidar2lidar/kinematic_transformer.h"
#include <ros/console.h>
#include <cmath>

namespace lidar2lidar {

KinematicTransformer::KinematicTransformer() {
    current_state_.valid = false;
}

void KinematicTransformer::updateKinematics(const std::vector<double>& dh_params) {
    if (dh_params.empty()) {
        ROS_WARN("KinematicTransformer: Empty DH parameters");
        return;
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // DH transformation (specific to excavator boom)
    Eigen::Matrix4f boom_angle_DH_mat = makeDHMatrix(0, 0.0, 0, -dh_params[0]);
    Eigen::Matrix4f boom_len_DH_mat = makeDHMatrix(0, 5.7, 0, 0.0);
    
    // Roll transformation (90 degrees)
    Eigen::Matrix4f roll_mat;
    roll_mat << 1, 0, 0, 0,
                0, std::cos(M_PI/2.0), -std::sin(M_PI/2.0), 0,
                0, std::sin(M_PI/2.0),  std::cos(M_PI/2.0), 0,
                0, 0, 0, 1;
    
    Eigen::Matrix4f origin = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f roll_transform = origin * roll_mat;
    Eigen::Matrix4f boom_yaw = roll_transform * boom_angle_DH_mat;
    Eigen::Matrix4f boom_pos = boom_yaw * boom_len_DH_mat;
    
    // Update state with all intermediate transforms
    current_state_.joint1.x = 0.0;
    current_state_.joint1.y = 0.0;
    current_state_.joint1.z = 0.0;
    
    current_state_.joint2.x = boom_pos(0, 3);
    current_state_.joint2.y = boom_pos(1, 3);
    current_state_.joint2.z = boom_pos(2, 3);
    
    current_state_.roll_transform = roll_transform;
    current_state_.boom_yaw = boom_yaw;
    current_state_.transform = boom_pos;
    current_state_.valid = true;
}

CloudPtr KinematicTransformer::transformCloud(
    const RawCloudPtr& input,
    double len_ratio,
    const Eigen::Vector3f& manual_offset,
    const Eigen::Vector3f& manual_rotation_rpy,
    const Eigen::Vector3f& additional_rotation_rpy) {
    
    CloudPtr output(new Cloud());
    
    if (!input || input->empty()) {
        ROS_WARN("KinematicTransformer: Empty input cloud");
        return output;
    }
    
    KinematicState state = getState();
    if (!state.valid) {
        ROS_WARN("KinematicTransformer: Invalid kinematic state");
        return output;
    }
    
    // Get midpoint with manual offset
    Eigen::Vector3f mid = getMidpoint(len_ratio, manual_offset);
    
    // Build orthonormal basis from joints
    Eigen::Vector3f x_axis(
        state.joint2.x - state.joint1.x,
        state.joint2.y - state.joint1.y,
        state.joint2.z - state.joint1.z
    );
    x_axis.normalize();
    
    Eigen::Matrix3f base_rotation = buildOrthonormalBasis(x_axis);
    
    // Apply manual rotation
    Eigen::Matrix3f manual_rot = rpyToRotationMatrix(manual_rotation_rpy);
    Eigen::Matrix3f combined_rotation = base_rotation * manual_rot;
    
    // First transformation: base + manual rotation
    CloudPtr intermediate(new Cloud());
    for (const auto& pt : input->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f p_transformed = combined_rotation * p + mid;
        
        CloudPoint p_out;
        p_out.x = p_transformed.x();
        p_out.y = p_transformed.y();
        p_out.z = p_transformed.z();
        p_out.intensity = pt.intensity;
        p_out.curvature = static_cast<float>(pt.offset_time);
        p_out.normal_x = 0.0f;
        p_out.normal_y = 0.0f;
        p_out.normal_z = 0.0f;
        
        intermediate->points.push_back(p_out);
    }
    
    // Second transformation: additional rotation
    Eigen::Matrix3f additional_rot = rpyToRotationMatrix(additional_rotation_rpy);
    
    for (const auto& pt : intermediate->points) {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        
        // Filter out points too close to origin
        if (p.norm() < 3.0) {
            continue;
        }
        
        Eigen::Vector3f p_rotated = additional_rot * p;
        
        CloudPoint p_out;
        p_out.x = p_rotated.x();
        p_out.y = p_rotated.y();
        p_out.z = p_rotated.z();
        p_out.intensity = pt.intensity;
        p_out.curvature = pt.curvature;
        p_out.normal_x = 0.0f;
        p_out.normal_y = 0.0f;
        p_out.normal_z = 0.0f;
        
        output->points.push_back(p_out);
    }
    
    return output;
}

KinematicTransformer::KinematicState KinematicTransformer::getState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_;
}

Eigen::Vector3f KinematicTransformer::getMidpoint(
    double ratio, 
    const Eigen::Vector3f& manual_offset) const {
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    Eigen::Vector3f mid;
    mid.x() = ratio * current_state_.joint1.x + (1.0 - ratio) * current_state_.joint2.x + manual_offset.x();
    mid.y() = ratio * current_state_.joint1.y + (1.0 - ratio) * current_state_.joint2.y + manual_offset.y();
    mid.z() = ratio * current_state_.joint1.z + (1.0 - ratio) * current_state_.joint2.z + manual_offset.z();
    
    return mid;
}

bool KinematicTransformer::isReady() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_state_.valid;
}

Eigen::Matrix4f KinematicTransformer::makeDHMatrix(
    float theta, float d, float a, float alpha) const {
    
    Eigen::Matrix4f mat;
    mat << std::cos(theta), -std::sin(theta) * std::cos(alpha),  std::sin(theta) * std::sin(alpha), a * std::cos(theta),
           std::sin(theta),  std::cos(theta) * std::cos(alpha), -std::cos(theta) * std::sin(alpha), a * std::sin(theta),
           0,                std::sin(alpha),                     std::cos(alpha),                    d,
           0,                0,                                   0,                                  1;
    return mat;
}

Eigen::Matrix3f KinematicTransformer::buildOrthonormalBasis(
    const Eigen::Vector3f& x_axis) const {
    
    Eigen::Vector3f x = x_axis.normalized();
    
    // Choose temporary up vector
    Eigen::Vector3f temp_up(0.0, 0.0, 1.0);
    if (std::fabs(x.dot(temp_up)) > 0.99) {
        temp_up = Eigen::Vector3f(0.0, 1.0, 0.0);
    }
    
    // Build orthonormal basis
    Eigen::Vector3f z = x.cross(temp_up).normalized();
    Eigen::Vector3f y = z.cross(x).normalized();
    
    Eigen::Matrix3f rotation;
    rotation.col(0) = x;
    rotation.col(1) = y;
    rotation.col(2) = z;
    
    return rotation;
}

Eigen::Matrix3f KinematicTransformer::rpyToRotationMatrix(
    const Eigen::Vector3f& rpy) const {
    
    float roll = rpy.x();
    float pitch = rpy.y();
    float yaw = rpy.z();
    
    // Roll (X-axis rotation)
    Eigen::Matrix3f R_x;
    R_x << 1, 0, 0,
           0, std::cos(roll), -std::sin(roll),
           0, std::sin(roll),  std::cos(roll);
    
    // Pitch (Y-axis rotation)
    Eigen::Matrix3f R_y;
    R_y << std::cos(pitch), 0, std::sin(pitch),
           0, 1, 0,
          -std::sin(pitch), 0, std::cos(pitch);
    
    // Yaw (Z-axis rotation)
    Eigen::Matrix3f R_z;
    R_z << std::cos(yaw), -std::sin(yaw), 0,
           std::sin(yaw),  std::cos(yaw), 0,
           0, 0, 1;
    
    // Combined rotation: R = R_z * R_y * R_x
    return R_z * R_y * R_x;
}

} // namespace lidar2lidar
