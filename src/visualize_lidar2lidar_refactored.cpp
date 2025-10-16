#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>
#include <visualization_msgs/InteractiveMarker.h>
#include <visualization_msgs/InteractiveMarkerControl.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nano_gicp/nano_gicp.hpp>
#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/impl/lsq_registration_impl.hpp>
#include <nano_gicp/impl/nano_gicp_impl.hpp>

#include "lidar2lidar/types.h"
#include "lidar2lidar/point_cloud_buffer.h"
#include "lidar2lidar/kinematic_transformer.h"
#include "lidar2lidar/point_cloud_aligner.h"

#include <chrono>
#include <array>
#include <algorithm>
#include <limits>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <mutex>
#include <numeric>
#include <atomic>

using namespace lidar2lidar;

class KinematicChainVisualizer
{
public:
    enum class AlignTimerStep
    {
        PREP = 0,
        ROI,
        DOWNSAMPLE_LiDAR1,
        DOWNSAMPLE_LiDAR2,
        ICP,
        GICP,
        COUNT
    };

    static constexpr size_t kAlignTimerCount = static_cast<size_t>(AlignTimerStep::COUNT);

    struct AlignmentStages
    {
        CloudPtr target_roi;                 // Target after ROI cropping
        CloudPtr kinematics_cloud;           // After kinematic transformation
        CloudPtr coarse_icp_cloud;           // After coarse ICP
        CloudPtr fine_icp_cloud;             // After fine ICP
        CloudPtr gicp_cloud;                 // After GICP
        bool coarse_icp_success = false;
        bool fine_icp_success = false;
        bool gicp_success = false;
    };

    struct AlignStatistics
    {
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
        
        // Step timings
        double prep_time_us = 0.0;
        double roi_time_us = 0.0;
        double downsample1_time_us = 0.0;
        double downsample2_time_us = 0.0;
        double icp_time_us = 0.0;
        double gicp_time_us = 0.0;
        
        void reset() {
            source_points = 0;
            source_coarse_points = 0;
            source_fine_points = 0;
            target_points = 0;
            target_roi_points = 0;
            target_coarse_points = 0;
            target_fine_points = 0;
            coarse_icp_converged = false;
            fine_icp_converged = false;
            gicp_converged = false;
            coarse_icp_fitness = -1.0;
            fine_icp_fitness = -1.0;
            gicp_fitness = -1.0;
            translation_norm = 0.0;
            rotation_angle_deg = 0.0;
            pre_icp_time_us = 0.0;
            icp_processing_time_us = 0.0;
            prep_time_us = 0.0;
            roi_time_us = 0.0;
            downsample1_time_us = 0.0;
            downsample2_time_us = 0.0;
            icp_time_us = 0.0;
            gicp_time_us = 0.0;
        }
    };

    KinematicChainVisualizer() : user_q_(0, 0, 0, 1)
    {
        ros::NodeHandle nh("~");
        
        // Initialize parameters
        loadParameters(nh);
        
        // Setup subscribers 
        sub_ = nh.subscribe("/Kinematic/DH_Angle", 20000, &KinematicChainVisualizer::DH_callback, this);
        lidar1_sub_ = nh.subscribe("/ml_/pointcloud", 200000, &KinematicChainVisualizer::lidar1_callback, this);
        lidar2_sub_ = nh.subscribe("/ml_/pointcloud2", 200000, &KinematicChainVisualizer::lidar2_callback, this);
        
        // Setup publishers
        pub_ = nh.advertise<visualization_msgs::MarkerArray>("/kinematic_chain_markers", 1);
        cloud1_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar1_transformed", 1);
        cloud1_rgb_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar1_transformed_rgb", 1);
        cloud1_roi_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar1_cropped_ROI", 1);
        cloud2_kinematics_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar2_kinematics", 1);
        cloud2_coarse_icp_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar2_coarse_ICP", 1);
        cloud2_fine_icp_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar2_fine_ICP", 1);
        cloud2_gicp_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar2_GICP", 1);
        cloud2_transformed_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/lidar2_transformed", 1);
        cloud_merged_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_merged", 1);
        cloud_merged_rgb_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_merged_rgb", 1);
        
        // Initialize buffers 
        lidar1_buffer_.reset(new PointCloudBuffer<CloudPoint>(0));  // 0 = unlimited
        lidar2_buffer_.reset(new PointCloudBuffer<RawPoint>(0));     // 0 = unlimited
        dh_buffer_.reset(new PointCloudBuffer<CloudPoint>(0));       // 0 = unlimited
        
        // Initialize transformer and aligner
        transformer_.reset(new KinematicTransformer());
        aligner_.reset(new PointCloudAligner());
        
        // Configure aligner
        configureAligner();
        
        // Interactive marker for manual control
        server_.reset(new interactive_markers::InteractiveMarkerServer("lidar_orientation_control"));
        createInteractiveMarker();
        
        // Processing timer
        processing_timer_ = nh.createTimer(ros::Duration(0.05), &KinematicChainVisualizer::processingTimerCallback, this);
        user_input_timer_ = nh.createTimer(ros::Duration(0.1), &KinematicChainVisualizer::inputWatcher, this);
    }

    ~KinematicChainVisualizer()
    {
        printTimingSummary();
    }

private:
    using HighResClock = std::chrono::high_resolution_clock;

    // ROS publishers and subscribers
    ros::Subscriber sub_, lidar2_sub_, lidar1_sub_;
    ros::Publisher pub_, cloud1_pub_, cloud1_rgb_pub_, cloud1_roi_pub_, cloud_merged_pub_, cloud_merged_rgb_pub_;
    ros::Publisher cloud2_kinematics_pub_, cloud2_coarse_icp_pub_, cloud2_fine_icp_pub_, cloud2_gicp_pub_;
    ros::Publisher cloud2_transformed_pub_;  // Final lidar2 result (with original fields)
    ros::Timer processing_timer_, user_input_timer_;
    std::string frame_id_ = "world";
    
    // Buffers (Stage 1: Data in buffers)
    std::shared_ptr<PointCloudBuffer<CloudPoint>> lidar1_buffer_;
    std::shared_ptr<PointCloudBuffer<RawPoint>> lidar2_buffer_;
    std::shared_ptr<PointCloudBuffer<CloudPoint>> dh_buffer_;
    
    // Processing modules (Stage 2: Separated functions)
    std::shared_ptr<KinematicTransformer> transformer_;
    std::shared_ptr<PointCloudAligner> aligner_;
    
    // Interactive marker
    std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
    
    // Parameters
    double manual_rotate_roll_, manual_rotate_pitch_, manual_rotate_yaw_;
    double additional_rotate_roll_, additional_rotate_pitch_, additional_rotate_yaw_;
    tf2::Quaternion user_q_;
    bool use_marker_ = false;
    geometry_msgs::Point joint1_, joint2_;
    double manual_len_ratio_ = 0.5;
    std::vector<double> lidar1_ext_t_, lidar1_ext_q_;
    double manual_x_add_ = 0.0, manual_y_add_ = 0.0, manual_z_add_ = 0.0;
    double icp_threshold_;
    double icp_delta_threshold_;
    
    // Improved pipeline parameters
    bool improve_pipeline_enabled_;
    bool improve_use_roi_;
    bool improve_use_dynamic_roi_;
    ROI static_roi_;
    double improve_roi_pad_x_, improve_roi_pad_y_, improve_roi_pad_z_;
    double improve_coarse_leaf_size_;
    double improve_fine_leaf_size_;
    int improve_coarse_max_iterations_;
    int improve_fine_max_iterations_;
    double improve_coarse_max_corr_dist_;
    double improve_fine_max_corr_dist_;
    double improve_icp_transformation_epsilon_;
    double improve_icp_euclidean_epsilon_;
    double improve_gicp_max_corr_dist_;
    int improve_gicp_max_iterations_;
    int improve_gicp_correspondence_randomness_;
    double improve_gicp_ransac_iterations_;
    double improve_gicp_transformation_epsilon_;
    double improve_gicp_euclidean_fitness_epsilon_;
    
    // State tracking (Stage 3: Mutex protected)
    mutable std::mutex state_mutex_;
    bool joints_ready_ = false;
    Eigen::Vector3f prev_icp_trans_;
    std::mutex lidar1_mutex_;
    std_msgs::Header lidar1_header_;
    Eigen::Matrix3f last_icp_rot_;
    Eigen::Vector3f last_icp_trans_;
    bool icp_first_success_ = false;
    
    // Timing statistics
    std::mutex timing_mutex_;
    std::vector<double> pre_icp_history_ms_;
    std::vector<double> icp_history_ms_;
    std::atomic<bool> stats_printed_{false};
    
    // Current metrics
    AlignStatistics current_metrics_;
    std::array<double, kAlignTimerCount> align_step_durations_;
    
    // ===== Stage 1: Callbacks only store data in buffers =====
    
    void DH_callback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        // Update kinematic transformer
        transformer_->updateKinematics(msg->data);
        
        // Update joint positions
        auto state = transformer_->getState();
        if (state.valid) {
            joint1_.x = state.joint1.x;
            joint1_.y = state.joint1.y;
            joint1_.z = state.joint1.z;
            
            joint2_.x = state.joint2.x;
            joint2_.y = state.joint2.y;
            joint2_.z = state.joint2.z;
            
            joints_ready_ = true;
            
            // Visualize markers
            visualizeKinematicChain(state);
        }
    }

    void lidar1_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // Stage 1: Only receive and store data
        RawCloudPtr cloud_in(new RawCloud());
        pcl::fromROSMsg(*msg, *cloud_in);
        
        // Apply LiDAR1 extrinsic transformation
        CloudPtr transformed = applyLidar1Extrinsic(cloud_in);
        
        // Store in buffer
        lidar1_buffer_->push(transformed, msg->header.stamp, msg->header);
        
        // Store header (mutex protected)
        {
            std::lock_guard<std::mutex> lock(lidar1_mutex_);
            lidar1_header_ = msg->header;
        }
        
        // Publish LiDAR1 transformed
        sensor_msgs::PointCloud2 cloud_out;
        pcl::toROSMsg(*transformed, cloud_out);
        cloud_out.header.frame_id = "world";
        cloud_out.header.stamp = msg->header.stamp;
        cloud1_pub_.publish(cloud_out);
    }

    void lidar2_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // Stage 1: Only receive and store data
        RawCloudPtr cloud_in(new RawCloud());
        pcl::fromROSMsg(*msg, *cloud_in);
        
        // Store in buffer
        lidar2_buffer_->push(cloud_in, msg->header.stamp, msg->header);
    }
    
    // ===== Stage 2: Separate processing functions =====
    
    void processingTimerCallback(const ros::TimerEvent& event)
    {   

        // Check if we have required data
        if (!joints_ready_) return;
        if (lidar1_buffer_->empty() || lidar2_buffer_->empty()) return;
        
        // Get front data and remove from buffer 
        typename PointCloudBuffer<CloudPoint>::BufferedData lidar1_data;
        typename PointCloudBuffer<RawPoint>::BufferedData lidar2_data;
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!lidar1_buffer_->getFront(lidar1_data)) return;
            if (!lidar2_buffer_->getFront(lidar2_data)) return;
        }
        
        CloudPtr lidar1_cloud = lidar1_data.cloud;
        ros::Time lidar1_time = lidar1_data.timestamp;
        RawCloudPtr lidar2_cloud = lidar2_data.cloud;
        ros::Time lidar2_time = lidar2_data.timestamp;
        // TODO timestamp validation

        // Process alignment
        processAlignment(lidar1_cloud, lidar2_cloud, lidar2_time);
        
        // Remove processed data from buffers
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            lidar1_buffer_->popFront();
            lidar2_buffer_->popFront();
        }
        
        // Log buffer status
        static int process_count = 0;
        if (++process_count % 50 == 0) {
            ROS_INFO("Buffer status - LiDAR1: %zu, LiDAR2: %zu, DH: %zu", 
                     lidar1_buffer_->size(), lidar2_buffer_->size(), dh_buffer_->size());
        }
    }
    
    void processAlignment(const CloudPtr& lidar1, const RawCloudPtr& lidar2, const ros::Time& stamp)
    {
        auto start_pre = HighResClock::now();
        
        // Stage 2: Function call for kinematic transformation
        CloudPtr transformed_lidar2 = applyKinematicTransform(lidar2);
        
        if (!transformed_lidar2 || transformed_lidar2->empty()) {
            ROS_WARN_THROTTLE(1.0, "Kinematic transformation failed");
            return;
        }
        
        auto end_pre = HighResClock::now();
        double pre_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_pre - start_pre).count() / 1000.0;
        
        // Stage 2: Function call for ICP/GICP alignment
        auto start_icp = HighResClock::now();
        
        AlignmentStages stages;
        stages.kinematics_cloud = transformed_lidar2;  // Save kinematics-only result
        
        if (improve_pipeline_enabled_) {
            stages = performImprovedAlignment(lidar1, transformed_lidar2);
            stages.kinematics_cloud = transformed_lidar2;  // Keep kinematics result
        } else {
            // Basic alignment doesn't have stages
            AlignmentResult result = performBasicAlignment(lidar1, transformed_lidar2);
            if (result.success) {
                stages.gicp_cloud = result.aligned_cloud;
                stages.gicp_success = true;
            }
        }
        
        auto end_icp = HighResClock::now();
        double icp_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_icp - start_icp).count() / 1000.0;
        
        // Update timing statistics (Stage 3: Mutex protected)
        {
            std::lock_guard<std::mutex> lock(timing_mutex_);
            pre_icp_history_ms_.push_back(pre_ms);
            icp_history_ms_.push_back(icp_ms);
        }
        
        // Print real-time status (like original version)
        current_metrics_.pre_icp_time_us = pre_ms * 1000.0;
        current_metrics_.icp_processing_time_us = icp_ms * 1000.0;
        printRealtimeStatus();
        
        // Stage 2: Function call for publishing results
        publishResults(lidar1, stages, stamp);
    }
    
    // ===== Stage 2: Individual function implementations =====
    
    CloudPtr applyLidar1Extrinsic(const RawCloudPtr& input)
    {
        CloudPtr output(new Cloud());
        
        // Convert quaternion to rotation matrix
        Eigen::Quaternionf quat(lidar1_ext_q_[3], lidar1_ext_q_[0], lidar1_ext_q_[1], lidar1_ext_q_[2]);
        quat.normalize();
        Eigen::Matrix3f rot = quat.toRotationMatrix();
        
        // Translation vector
        Eigen::Vector3f trans(lidar1_ext_t_[0], lidar1_ext_t_[1], lidar1_ext_t_[2]);
        
        // Transform each point
        for (const auto& pt : input->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_transformed = rot * p + trans;
            
            CloudPoint p_out;
            p_out.x = p_transformed.x();
            p_out.y = p_transformed.y();
            p_out.z = p_transformed.z();
            p_out.intensity = pt.intensity;
            p_out.curvature = static_cast<float>(pt.offset_time);
            p_out.normal_x = 0.0f;
            p_out.normal_y = 0.0f;
            p_out.normal_z = 0.0f;
            
            output->points.push_back(p_out);
        }
        
        return output;
    }
    
    CloudPtr applyKinematicTransform(const RawCloudPtr& input)
    {
        // Manual rotation and offset
        Eigen::Vector3f manual_offset(manual_x_add_, manual_y_add_, manual_z_add_);
        
        // Convert degrees to radians (CRITICAL FIX!)
        Eigen::Vector3f manual_rotation(
            manual_rotate_roll_ * M_PI / 180.0,
            manual_rotate_pitch_ * M_PI / 180.0,
            manual_rotate_yaw_ * M_PI / 180.0
        );
        Eigen::Vector3f additional_rotation(
            additional_rotate_roll_ * M_PI / 180.0,
            additional_rotate_pitch_ * M_PI / 180.0,
            additional_rotate_yaw_ * M_PI / 180.0
        );
        
        // Apply transformation
        CloudPtr transformed = transformer_->transformCloud(
            input,
            manual_len_ratio_,
            manual_offset,
            manual_rotation,
            additional_rotation
        );
        
        return transformed;
    }
    

    //TODO
    AlignmentResult performBasicAlignment(const CloudPtr& target, const CloudPtr& source)
    {
        // Basic alignment (legacy method)
        AlignmentResult result;
        
        // Just return initial guess for now
        result.aligned_cloud = source;
        result.success = true;
        
        return result;
    }
    
    AlignmentStages performImprovedAlignment(const CloudPtr& target, const CloudPtr& source)
    {
        using TimerIdx = AlignTimerStep;
        std::fill(align_step_durations_.begin(), align_step_durations_.end(), 0.0);
        
        auto t_start = HighResClock::now();
        AlignmentStages stages;
        
        // PREP: Initialize metrics
        current_metrics_.reset();
        current_metrics_.source_points = source->size();
        current_metrics_.target_points = target->size();
        auto t_prep = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::PREP)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_prep - t_start).count();
        
        // ROI: Crop target
        CloudPtr target_roi = target;
        if (improve_use_roi_) {
            if (improve_use_dynamic_roi_) {
                auto state = transformer_->getState();
                Eigen::Vector3f roi_center = transformer_->getMidpoint(manual_len_ratio_, Eigen::Vector3f::Zero());
                target_roi = aligner_->cropWithDynamicROI(
                    target,
                    roi_center,
                    static_roi_.max.x() - static_roi_.min.x() + 2 * improve_roi_pad_x_,
                    static_roi_.max.y() - static_roi_.min.y() + 2 * improve_roi_pad_y_,
                    static_roi_.max.z() - static_roi_.min.z() + 2 * improve_roi_pad_z_
                );
            } else {
                target_roi = aligner_->cropWithStaticROI(target, static_roi_);
            }
        }
        stages.target_roi = target_roi;  // Save ROI result
        current_metrics_.target_roi_points = target_roi->size();
        auto t_roi = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::ROI)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_roi - t_prep).count();
        
        // DOWNSAMPLE1: Coarse downsampling
        CloudPtr source_coarse = aligner_->downsample(source, 
            improve_coarse_leaf_size_, improve_coarse_leaf_size_, improve_coarse_leaf_size_);
        CloudPtr target_coarse = aligner_->downsample(target_roi,
            improve_coarse_leaf_size_, improve_coarse_leaf_size_, improve_coarse_leaf_size_);
        current_metrics_.source_coarse_points = source_coarse->size();
        current_metrics_.target_coarse_points = target_coarse->size();
        auto t_ds1 = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::DOWNSAMPLE_LiDAR1)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_ds1 - t_roi).count();
        
        // ICP: Coarse ICP
        auto t_icp_start = HighResClock::now();
        AlignmentResult coarse_result = aligner_->alignICP(
            source_coarse,
            target_coarse,
            Eigen::Matrix4f::Identity(),
            improve_coarse_max_iterations_,
            improve_icp_transformation_epsilon_,
            improve_icp_euclidean_epsilon_,
            improve_coarse_max_corr_dist_  // CRITICAL: max correspondence distance
        );
        
        if (!coarse_result.success) {
            ROS_WARN_THROTTLE(1.0, "Coarse ICP failed to converge");
            return stages;
        }
        
        // Transform original source with coarse ICP result
        stages.coarse_icp_cloud.reset(new Cloud());
        pcl::transformPointCloud(*source, *stages.coarse_icp_cloud, coarse_result.final_transform);
        stages.coarse_icp_success = true;
        
        current_metrics_.coarse_icp_converged = coarse_result.success;
        current_metrics_.coarse_icp_fitness = coarse_result.fitness_score;
        
        ROS_INFO_STREAM("[IMPROVE] Coarse ICP converged, fitness=" << coarse_result.fitness_score 
                        << " (source=" << coarse_result.source_points << ", target=" << coarse_result.target_points << ")");
        
        // DOWNSAMPLE2: Fine downsampling
        CloudPtr source_fine = aligner_->downsample(source,
            improve_fine_leaf_size_, improve_fine_leaf_size_, improve_fine_leaf_size_);
        CloudPtr target_fine = aligner_->downsample(target_roi,
            improve_fine_leaf_size_, improve_fine_leaf_size_, improve_fine_leaf_size_);
        current_metrics_.source_fine_points = source_fine->size();
        current_metrics_.target_fine_points = target_fine->size();
        auto t_ds2 = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::DOWNSAMPLE_LiDAR2)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_ds2 - t_ds1).count();
        
        // Fine ICP
        AlignmentResult fine_result = aligner_->alignICP(
            source_fine,
            target_fine,
            coarse_result.final_transform,
            improve_fine_max_iterations_,
            improve_icp_transformation_epsilon_,
            improve_icp_euclidean_epsilon_,
            improve_fine_max_corr_dist_  // CRITICAL: max correspondence distance
        );
        
        if (!fine_result.success) {
            ROS_WARN_THROTTLE(1.0, "Fine ICP failed to converge");
            return stages;
        }
        
        // Transform original source with fine ICP result
        stages.fine_icp_cloud.reset(new Cloud());
        pcl::transformPointCloud(*source, *stages.fine_icp_cloud, fine_result.final_transform);
        stages.fine_icp_success = true;
        
        current_metrics_.fine_icp_converged = fine_result.success;
        current_metrics_.fine_icp_fitness = fine_result.fitness_score;
        
        ROS_INFO_STREAM("[IMPROVE] Fine ICP converged, fitness=" << fine_result.fitness_score
                        << " (source=" << fine_result.source_points << ", target=" << fine_result.target_points << ")");
        
        auto t_icp_end = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::ICP)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_icp_end - t_icp_start).count();
        
        // GICP: Refinement
        auto t_gicp_start = HighResClock::now();
        AlignmentResult gicp_result = aligner_->alignGICP(
            source_fine,
            target_fine,
            fine_result.final_transform,
            improve_gicp_max_iterations_,
            improve_gicp_transformation_epsilon_
        );
        
        // Transform original source with GICP result
        stages.gicp_cloud.reset(new Cloud());
        pcl::transformPointCloud(*source, *stages.gicp_cloud, gicp_result.final_transform);
        stages.gicp_success = gicp_result.success;
        
        current_metrics_.gicp_converged = gicp_result.success;
        current_metrics_.gicp_fitness = gicp_result.fitness_score;
        
        // Calculate transform metrics
        Eigen::Vector3f translation = gicp_result.final_transform.block<3,1>(0,3);
        current_metrics_.translation_norm = translation.norm();
        Eigen::Matrix3f rotation = gicp_result.final_transform.block<3,3>(0,0);
        Eigen::AngleAxisf angle_axis(rotation);
        current_metrics_.rotation_angle_deg = angle_axis.angle() * 180.0 / M_PI;
        
        ROS_INFO_STREAM("[IMPROVE] GICP converged, fitness=" << gicp_result.fitness_score
                        << " (source=" << gicp_result.source_points << ", target=" << gicp_result.target_points << ")");
        
        auto t_gicp_end = HighResClock::now();
        align_step_durations_[static_cast<size_t>(TimerIdx::GICP)] = 
            std::chrono::duration_cast<std::chrono::microseconds>(t_gicp_end - t_gicp_start).count();
        
        // Copy timing to metrics
        current_metrics_.prep_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::PREP)];
        current_metrics_.roi_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::ROI)];
        current_metrics_.downsample1_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::DOWNSAMPLE_LiDAR1)];
        current_metrics_.downsample2_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::DOWNSAMPLE_LiDAR2)];
        current_metrics_.icp_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::ICP)];
        current_metrics_.gicp_time_us = align_step_durations_[static_cast<size_t>(TimerIdx::GICP)];
        
        ROS_DEBUG("Alignment took %ld microseconds", 
            std::chrono::duration_cast<std::chrono::microseconds>(t_gicp_end - t_start).count());
        
        return stages;
    }
    
    void publishResults(const CloudPtr& lidar1, const AlignmentStages& stages, const ros::Time& stamp)
    {
        // Helper function to convert to RGB point cloud with specific color
        auto toColoredCloud = [](const CloudPtr& input, uint8_t r, uint8_t g, uint8_t b) 
            -> pcl::PointCloud<pcl::PointXYZRGB>::Ptr {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
            for (const auto& pt : input->points) {
                pcl::PointXYZRGB p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                p.r = r;
                p.g = g;
                p.b = b;
                colored->points.push_back(p);
            }
            colored->width = colored->points.size();
            colored->height = 1;
            colored->is_dense = false;
            return colored;
        };
        
        // 1. Publish /lidar1_transformed (original fields: x,y,z,intensity,curvature,normal)
        if (lidar1 && !lidar1->empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*lidar1, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud1_pub_.publish(cloud_msg);
        }
        
        // 2. Publish /lidar1_transformed_rgb - 빨강 (Red: 255, 0, 0)
        if (lidar1 && !lidar1->empty()) {
            auto lidar1_colored = toColoredCloud(lidar1, 255, 0, 0);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*lidar1_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud1_rgb_pub_.publish(cloud_msg);
        }
        
        // 3. Publish /lidar1_cropped_ROI - 주황 (Orange: 255, 165, 0)
        if (stages.target_roi && !stages.target_roi->empty()) {
            auto lidar1_roi_colored = toColoredCloud(stages.target_roi, 255, 165, 0);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*lidar1_roi_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud1_roi_pub_.publish(cloud_msg);
        }
        
        // 4. Publish /lidar2_kinematics - 노랑 (Yellow: 255, 255, 0)
        if (stages.kinematics_cloud && !stages.kinematics_cloud->empty()) {
            auto kinematics_colored = toColoredCloud(stages.kinematics_cloud, 255, 255, 0);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*kinematics_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud2_kinematics_pub_.publish(cloud_msg);
        }
        
        // 5. Publish /lidar2_coarse_ICP - 초록 (Green: 0, 255, 0)
        if (stages.coarse_icp_cloud && !stages.coarse_icp_cloud->empty()) {
            auto coarse_colored = toColoredCloud(stages.coarse_icp_cloud, 0, 255, 0);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*coarse_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud2_coarse_icp_pub_.publish(cloud_msg);
        }
        
        // 6. Publish /lidar2_fine_ICP - 파랑 (Blue: 0, 0, 255)
        if (stages.fine_icp_cloud && !stages.fine_icp_cloud->empty()) {
            auto fine_colored = toColoredCloud(stages.fine_icp_cloud, 0, 0, 255);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*fine_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud2_fine_icp_pub_.publish(cloud_msg);
        }
        
        // 7. Publish /lidar2_GICP - 남색 (Indigo: 75, 0, 130)
        if (stages.gicp_cloud && !stages.gicp_cloud->empty()) {
            auto gicp_colored = toColoredCloud(stages.gicp_cloud, 75, 0, 130);
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*gicp_colored, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud2_gicp_pub_.publish(cloud_msg);
        }
        
        // 8. Publish /lidar2_transformed (final stage with original fields: x,y,z,intensity,curvature,normal)
        CloudPtr final_lidar2_original;
        
        if (stages.gicp_success && stages.gicp_cloud) {
            final_lidar2_original = stages.gicp_cloud;
        } else if (stages.fine_icp_success && stages.fine_icp_cloud) {
            final_lidar2_original = stages.fine_icp_cloud;
        } else if (stages.coarse_icp_success && stages.coarse_icp_cloud) {
            final_lidar2_original = stages.coarse_icp_cloud;
        } else if (stages.kinematics_cloud) {
            final_lidar2_original = stages.kinematics_cloud;
        }
        
        if (final_lidar2_original && !final_lidar2_original->empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*final_lidar2_original, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud2_transformed_pub_.publish(cloud_msg);
        }
        
        // 9. Publish /cloud_merged (lidar1 + lidar2_transformed with original fields)
        CloudPtr merged(new Cloud());
        
        // Add lidar1 (with original fields)
        if (lidar1 && !lidar1->empty()) {
            *merged += *lidar1;
        }
        
        // Add final lidar2 (with original fields)
        if (final_lidar2_original && !final_lidar2_original->empty()) {
            *merged += *final_lidar2_original;
        }
        
        // Set cloud properties
        merged->width = merged->points.size();
        merged->height = 1;
        merged->is_dense = false;
        
        // Publish merged cloud (original fields)
        if (!merged->empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*merged, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud_merged_pub_.publish(cloud_msg);
        }
        
        // 10. Publish /cloud_merged_rgb (for better visualization with distinct colors)
        // LiDAR1 = Cyan (0,255,255), LiDAR2 = Magenta (255,0,255)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
        
        // Add lidar1 as Cyan
        if (lidar1 && !lidar1->empty()) {
            for (const auto& pt : lidar1->points) {
                pcl::PointXYZRGB p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                p.r = 0;
                p.g = 255;
                p.b = 255;
                merged_rgb->points.push_back(p);
            }
        }
        
        // Add final lidar2 as Magenta
        if (final_lidar2_original && !final_lidar2_original->empty()) {
            for (const auto& pt : final_lidar2_original->points) {
                pcl::PointXYZRGB p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                p.r = 255;
                p.g = 0;
                p.b = 255;
                merged_rgb->points.push_back(p);
            }
        }
        
        // Set cloud properties
        merged_rgb->width = merged_rgb->points.size();
        merged_rgb->height = 1;
        merged_rgb->is_dense = false;
        
        // Publish merged RGB cloud
        if (!merged_rgb->empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(*merged_rgb, cloud_msg);
            cloud_msg.header.frame_id = "world";
            cloud_msg.header.stamp = stamp;
            cloud_merged_rgb_pub_.publish(cloud_msg);
        }
        
        // Update state (mutex protected) - use the final transform
        if (stages.gicp_success) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            // Extract transform from GICP result
            // Note: We don't have the transform matrix in stages, so we skip this for now
            // This would need to be added to AlignmentStages if needed
            icp_first_success_ = true;
        }
    }
    
    // ===== Helper functions =====
    
    void loadParameters(ros::NodeHandle& nh)
    {
        nh.param<bool>("use_marker_", use_marker_, false);
        nh.param<double>("manual_len_ratio_", manual_len_ratio_, 0.5);
        nh.param<double>("manual_rotate_roll_", manual_rotate_roll_, 0.0);
        nh.param<double>("manual_rotate_pitch_", manual_rotate_pitch_, 0.0);
        nh.param<double>("manual_rotate_yaw_", manual_rotate_yaw_, 0.0);
        nh.param<double>("additional_rotate_roll_", additional_rotate_roll_, 0.0);
        nh.param<double>("additional_rotate_pitch_", additional_rotate_pitch_, 0.0);
        nh.param<double>("additional_rotate_yaw_", additional_rotate_yaw_, 0.0);
        nh.param<double>("manual_x_add_", manual_x_add_, 0.0);
        nh.param<double>("manual_y_add_", manual_y_add_, 0.0);
        nh.param<double>("manual_z_add_", manual_z_add_, 0.0);
        nh.param<double>("icp_threshold_", icp_threshold_, 3.5);
        nh.param<double>("icp_delta_threshold_", icp_delta_threshold_, 0.3);

        // Load LiDAR1 extrinsic
        if (!nh.getParam("lidar1_ext_t_", lidar1_ext_t_)) {
            ROS_WARN("Parameter 'lidar1_ext_t_' not found, using default: [0.0, 0.0, 0.0]");
            lidar1_ext_t_ = {0.0, 0.0, 0.0};
        }
        if (!nh.getParam("lidar1_ext_q_", lidar1_ext_q_)) {
            ROS_WARN("Parameter 'lidar1_ext_q_' not found, using default: [0.0, 0.0, 0.0, 1.0]");
            lidar1_ext_q_ = {0.0, 0.0, 0.0, 1.0};
        }

        // Improved pipeline parameters
        nh.param<bool>("use_improve_pipeline_", improve_pipeline_enabled_, false);
        nh.param<bool>("improve_use_dynamic_roi_", improve_use_dynamic_roi_, true);
        
        nh.param<double>("improve_roi_pad_x_", improve_roi_pad_x_, 0.5);
        nh.param<double>("improve_roi_pad_y_", improve_roi_pad_y_, 0.5);
        nh.param<double>("improve_roi_pad_z_", improve_roi_pad_z_, 0.5);
        nh.param<double>("improve_coarse_leaf_size_", improve_coarse_leaf_size_, 0.4);
        nh.param<double>("improve_fine_leaf_size_", improve_fine_leaf_size_, 0.15);
        nh.param<int>("improve_coarse_max_iterations_", improve_coarse_max_iterations_, 30);
        nh.param<int>("improve_fine_max_iterations_", improve_fine_max_iterations_, 20);
        nh.param<double>("improve_coarse_max_corr_dist_", improve_coarse_max_corr_dist_, 1.0);
        nh.param<double>("improve_fine_max_corr_dist_", improve_fine_max_corr_dist_, 0.5);
        nh.param<double>("improve_icp_transformation_epsilon_", improve_icp_transformation_epsilon_, 1e-5);
        nh.param<double>("improve_icp_euclidean_epsilon_", improve_icp_euclidean_epsilon_, 1e-5);
        nh.param<double>("improve_gicp_max_corr_dist_", improve_gicp_max_corr_dist_, 0.4);
        nh.param<int>("improve_gicp_max_iterations_", improve_gicp_max_iterations_, 15);
        nh.param<int>("improve_gicp_correspondence_randomness_", improve_gicp_correspondence_randomness_, 15);
        nh.param<double>("improve_gicp_ransac_iterations_", improve_gicp_ransac_iterations_, 50);
        nh.param<double>("improve_gicp_transformation_epsilon_", improve_gicp_transformation_epsilon_, 1e-5);
        nh.param<double>("improve_gicp_euclidean_fitness_epsilon_", improve_gicp_euclidean_fitness_epsilon_, 1e-5);

        // Initialize state
        prev_icp_trans_ = Eigen::Vector3f::Zero();
        last_icp_rot_ = Eigen::Matrix3f::Identity();
        last_icp_trans_ = Eigen::Vector3f::Zero();
    }
    
    void configureAligner()
    {
        aligner_->setICPMaxIterations(improve_coarse_max_iterations_);
        aligner_->setICPTransformationEpsilon(improve_icp_transformation_epsilon_);
        aligner_->setICPEuclideanFitnessEpsilon(improve_icp_euclidean_epsilon_);
        
        aligner_->setGICPMaxIterations(improve_gicp_max_iterations_);
        aligner_->setGICPTransformationEpsilon(improve_gicp_transformation_epsilon_);
        aligner_->setGICPCorrespondenceDistance(improve_gicp_max_corr_dist_);
    }
    
    void visualizeKinematicChain(const KinematicTransformer::KinematicState& state)
    {
        visualization_msgs::MarkerArray marker_array;
        ros::Time stamp = ros::Time::now();
        
        // Create axis markers for visualization (same as original)
        // Origin frame (identity) - Red axes
        create_axis_markers(Eigen::Matrix4f::Identity(), 0, "world", stamp, marker_array, 1.0, 0.0, 0.0);
        
        // After roll transform - Green axes
        create_axis_markers(state.roll_transform, 3, "world", stamp, marker_array, 0.0, 1.0, 0.0);
        
        // After boom yaw (roll + boom angle) - Blue axes
        create_axis_markers(state.boom_yaw, 6, "world", stamp, marker_array, 0.0, 0.0, 1.0);
        
        // Final boom position (roll + boom angle + boom length) - Blue axes
        create_axis_markers(state.transform, 9, "world", stamp, marker_array, 0.0, 0.0, 1.0);
        
        pub_.publish(marker_array);
    }
    
    void create_axis_markers(const Eigen::Matrix4f& transform, int id_offset, 
                           const std::string& frame_id, const ros::Time& stamp,
                           visualization_msgs::MarkerArray& marker_array,
                           float r, float g, float b)
    {
        // Extract rotation and translation
        Eigen::Matrix3f R = transform.block<3,3>(0,0);
        Eigen::Vector3f t = transform.block<3,1>(0,3);

        // Define axes (x, y, z)
        std::vector<Eigen::Vector3f> axes = {
            R * Eigen::Vector3f(0.5, 0, 0), // X-axis
            R * Eigen::Vector3f(0, 0.5, 0), // Y-axis
            R * Eigen::Vector3f(0, 0, 0.5)  // Z-axis
        };
        std::vector<std::tuple<float, float, float>> colors = {
            {r, 0, 0}, // Red for x
            {0, g, 0}, // Green for y
            {0, 0, b}  // Blue for z
        };

        for (int i = 0; i < 3; ++i) 
        {
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = stamp;
            marker.ns = "axes";
            marker.id = id_offset + i;
            marker.type = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;

            // Start point (origin of frame)
            marker.points.resize(2);
            marker.points[0].x = t(0);
            marker.points[0].y = t(1);
            marker.points[0].z = t(2);
            // End point (along axis)
            marker.points[1].x = t(0) + axes[i](0);
            marker.points[1].y = t(1) + axes[i](1);
            marker.points[1].z = t(2) + axes[i](2);

            marker.scale.x = 0.05; // Shaft diameter
            marker.scale.y = 0.1;  // Head diameter
            marker.scale.z = 0.1;  // Head length
            marker.color.a = 1.0;
            marker.color.r = std::get<0>(colors[i]);
            marker.color.g = std::get<1>(colors[i]);
            marker.color.b = std::get<2>(colors[i]);

            marker_array.markers.push_back(marker);
        }
    }
    
    void createInteractiveMarker()
    {
        visualization_msgs::InteractiveMarker int_marker;
        int_marker.header.frame_id = "world";
        int_marker.name = "lidar_orientation";
        int_marker.description = "RPY Control";
        int_marker.scale = 1.0;
        
        tf2::Quaternion q(0, 0, 0, 1);
        q.normalize();
        int_marker.pose.orientation = tf2::toMsg(q);

        addRotationControl(int_marker, "rotate_x", 1, 0, 0);
        addRotationControl(int_marker, "rotate_y", 0, 1, 0);
        addRotationControl(int_marker, "rotate_z", 0, 0, 1);

        server_->insert(int_marker, boost::bind(&KinematicChainVisualizer::processFeedback, this, _1));
        server_->applyChanges();
    }

    void addRotationControl(visualization_msgs::InteractiveMarker& marker, const std::string& name,
                           float x, float y, float z)
    {
        visualization_msgs::InteractiveMarkerControl control;
        control.orientation.w = 1;
        control.orientation.x = x;
        control.orientation.y = y;
        control.orientation.z = z;
        control.name = name;
        control.interaction_mode = visualization_msgs::InteractiveMarkerControl::ROTATE_AXIS;
        marker.controls.push_back(control);
    }

    void processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr& feedback)
    {
        tf2::Quaternion q;
        tf2::fromMsg(feedback->pose.orientation, q);
        q.normalize();
        user_q_ = q;
    }
    
    void inputWatcher(const ros::TimerEvent& event)
    {
        static bool prompt_printed = false;
        if (!prompt_printed) {
            ROS_INFO("Press 'q' then Enter to stop the node and print timing statistics.");
            prompt_printed = true;
        }
        
        std::streambuf* buf = std::cin.rdbuf();
        if (!buf || buf->in_avail() <= 0) {
            return;
        }
        
        std::string line;
        std::getline(std::cin, line);
        if (line == "q" || line == "Q" || line == "quit" || line == "exit") {
            printTimingSummary();
            ros::shutdown();
        }
    }
    
    void appendTimingStats(const std::string& label, const std::vector<double>& samples, std::ostringstream& out)
    {
        out << label << ":\n";
        if (samples.empty()) {
            out << "  samples : 0\n";
            return;
        }
        
        auto minmax = std::minmax_element(samples.begin(), samples.end());
        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        double mean = sum / static_cast<double>(samples.size());
        double variance = 0.0;
        for (double v : samples) {
            double diff = v - mean;
            variance += diff * diff;
        }
        if (samples.size() > 1) {
            variance /= static_cast<double>(samples.size() - 1);
        } else {
            variance = 0.0;
        }
        double stddev = std::sqrt(std::max(0.0, variance));
        
        out << std::fixed << std::setprecision(3)
            << "  samples : " << samples.size() << '\n'
            << "  mean    : " << mean << " ms\n"
            << "  min     : " << *minmax.first << " ms\n"
            << "  max     : " << *minmax.second << " ms\n"
            << "  stddev  : " << stddev << " ms\n";
    }
    
    void printRealtimeStatus()
    {
        std::ostringstream status;
        status << std::fixed << std::setprecision(3);
        
        // Alignment timers [us]
        status << "Alignment timers [us]\n";
        status << "  PREP        : " << current_metrics_.prep_time_us << '\n';
        status << "  ROI         : " << current_metrics_.roi_time_us << '\n';
        status << "  DOWNSAMPLE1 : " << current_metrics_.downsample1_time_us << '\n';
        status << "  DOWNSAMPLE2 : " << current_metrics_.downsample2_time_us << '\n';
        status << "  ICP         : " << current_metrics_.icp_time_us << '\n';
        status << "  GICP        : " << current_metrics_.gicp_time_us << '\n';
        status << '\n';
        
        // Alignment metrics
        status << "Alignment metrics\n";
        status << "  source            : " << current_metrics_.source_points << " pts\n";
        status << "  source_coarse     : " << current_metrics_.source_coarse_points << " pts\n";
        status << "  source_fine       : " << current_metrics_.source_fine_points << " pts\n";
        status << "  target            : " << current_metrics_.target_points << " pts\n";
        status << "  target_roi        : " << current_metrics_.target_roi_points << " pts\n";
        status << "  target_coarse     : " << current_metrics_.target_coarse_points << " pts\n";
        status << "  target_fine       : " << current_metrics_.target_fine_points << " pts\n";
        status << "  coarse_icp_conv   : " << (current_metrics_.coarse_icp_converged ? "true" : "false") << '\n';
        status << "  coarse_icp_fitness: " << current_metrics_.coarse_icp_fitness << '\n';
        status << "  fine_icp_conv     : " << (current_metrics_.fine_icp_converged ? "true" : "false") << '\n';
        status << "  fine_icp_fitness  : " << current_metrics_.fine_icp_fitness << '\n';
        status << "  gicp_conv         : " << (current_metrics_.gicp_converged ? "true" : "false") << '\n';
        status << "  gicp_fitness      : " << current_metrics_.gicp_fitness << '\n';
        status << "  translation       : " << current_metrics_.translation_norm << " m\n";
        status << "  rotation          : " << current_metrics_.rotation_angle_deg << " deg\n";
        status << "  pre_icp           : " << (current_metrics_.pre_icp_time_us / 1000.0) << " ms\n";
        status << "  icp               : " << (current_metrics_.icp_processing_time_us / 1000.0) << " ms\n";
        
        // Clear screen and move cursor to top-left, then print status
        std::cout << "\033[2J\033[H" << status.str() << std::flush;
    }
    
    void printTimingSummary()
    {
        if (stats_printed_.exchange(true)) {
            return;
        }
        
        std::vector<double> pre_samples;
        std::vector<double> icp_samples;
        {
            std::lock_guard<std::mutex> lock(timing_mutex_);
            pre_samples = pre_icp_history_ms_;
            icp_samples = icp_history_ms_;
        }
        
        std::ostringstream out;
        out << "\n==== Timing Summary (ms) ====\n";
        appendTimingStats("Pre-ICP processing", pre_samples, out);
        appendTimingStats("ICP + GICP processing", icp_samples, out);
        out << "=============================\n";
        std::cout << out.str() << std::flush;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinematic_chain_visualizer");
    KinematicChainVisualizer visualizer;
    ros::spin();
    return 0;
}
