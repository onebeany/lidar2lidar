#include "lidar2lidar/point_cloud_buffer.h"
#include <algorithm>
#include <ros/console.h>

namespace lidar2lidar {

template<typename PointT>
PointCloudBuffer<PointT>::PointCloudBuffer(size_t max_size) 
    : max_size_(max_size) {
    // max_size_ = 0 means unlimited buffer (like lvi_q)
    if (max_size_ == 0) {
        ROS_INFO("PointCloudBuffer: Unlimited buffer size");
    }
}

template<typename PointT>
void PointCloudBuffer<PointT>::push(
    const typename pcl::PointCloud<PointT>::Ptr& cloud,
    const ros::Time& stamp,
    const std_msgs::Header& header) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    BufferedData data;
    data.cloud = cloud;
    data.timestamp = stamp;
    data.header = header;
    data.valid = (cloud && !cloud->empty());
    
    buffer_.push_back(data);
    
    // Remove oldest if exceeds max size (only if max_size > 0)
    if (max_size_ > 0) {
        while (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }
    // If max_size_ == 0, keep all data (unlimited buffer like lvi_q)
}

template<typename PointT>
bool PointCloudBuffer<PointT>::getLatest(BufferedData& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (buffer_.empty()) {
        return false;
    }
    
    output = buffer_.back();
    return output.valid;
}

template<typename PointT>
bool PointCloudBuffer<PointT>::getByTime(
    const ros::Time& target_time, 
    BufferedData& output, 
    double tolerance) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (buffer_.empty()) {
        return false;
    }
    
    // Find closest timestamp
    auto closest = std::min_element(
        buffer_.begin(), 
        buffer_.end(),
        [&target_time](const BufferedData& a, const BufferedData& b) {
            return std::abs((a.timestamp - target_time).toSec()) < 
                   std::abs((b.timestamp - target_time).toSec());
        }
    );
    
    if (closest == buffer_.end()) {
        return false;
    }
    
    double time_diff = std::abs((closest->timestamp - target_time).toSec());
    if (time_diff > tolerance) {
        ROS_DEBUG("PointCloudBuffer: Time difference %.3f exceeds tolerance %.3f", 
                  time_diff, tolerance);
        return false;
    }
    
    output = *closest;
    return output.valid;
}

template<typename PointT>
bool PointCloudBuffer<PointT>::getFront(BufferedData& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (buffer_.empty()) {
        return false;
    }
    
    output = buffer_.front();
    return output.valid;
}

template<typename PointT>
void PointCloudBuffer<PointT>::popFront() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!buffer_.empty()) {
        buffer_.pop_front();
    }
}

template<typename PointT>
void PointCloudBuffer<PointT>::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
}

template<typename PointT>
size_t PointCloudBuffer<PointT>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

template<typename PointT>
bool PointCloudBuffer<PointT>::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

// Explicit template instantiation
template class PointCloudBuffer<pcl::PointXYZINormal>;
template class PointCloudBuffer<lidar2lidar::RawPoint>;

} // namespace lidar2lidar
