#ifndef LIDAR2LIDAR_POINT_CLOUD_BUFFER_H
#define LIDAR2LIDAR_POINT_CLOUD_BUFFER_H

#include <deque>
#include <mutex>
#include <ros/time.h>
#include <std_msgs/Header.h>
#include "lidar2lidar/types.h"

namespace lidar2lidar {

template<typename PointT>
class PointCloudBuffer {
public:
    struct BufferedData {
        typename pcl::PointCloud<PointT>::Ptr cloud;
        ros::Time timestamp;
        std_msgs::Header header;
        bool valid;
        
        BufferedData() 
            : cloud(new pcl::PointCloud<PointT>()), 
              valid(false) {}
    };
    
    explicit PointCloudBuffer(size_t max_size = 10);
    ~PointCloudBuffer() = default;
    
    // Thread-safe operations
    void push(const typename pcl::PointCloud<PointT>::Ptr& cloud, 
              const ros::Time& stamp,
              const std_msgs::Header& header);
    
    bool getLatest(BufferedData& output);
    bool getByTime(const ros::Time& target_time, BufferedData& output, double tolerance = 0.1);
    bool getFront(BufferedData& output);  // Get and remove front element (like lvi_q)
    void popFront();  // Remove front element
    
    void clear();
    size_t size() const;
    bool empty() const;
    
private:
    std::deque<BufferedData> buffer_;
    size_t max_size_;
    mutable std::mutex mutex_;
};

} // namespace lidar2lidar

#endif // LIDAR2LIDAR_POINT_CLOUD_BUFFER_H
