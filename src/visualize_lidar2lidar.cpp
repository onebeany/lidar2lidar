#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <interactive_markers/interactive_marker_server.h>
#include <interactive_markers/menu_handler.h>
#include <visualization_msgs/InteractiveMarker.h>
#include <visualization_msgs/InteractiveMarkerControl.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <Eigen/Geometry>
#include <cmath>
#include <pcl/registration/icp.h>
#include <nano_gicp/nano_gicp.hpp>
#include <nano_gicp/impl/lsq_registration_impl.hpp>
#include <nano_gicp/impl/nano_gicp_impl.hpp>
#include <chrono> // Add this header for timing

class KinematicChainVisualizer
{
public:
    KinematicChainVisualizer() : user_q_(0, 0, 0, 1)
    {
        ros::NodeHandle nh("~");
        sub_ = nh.subscribe("/Kinematic/DH_Angle", 1, &KinematicChainVisualizer::DH_callback, this);
        lidar1_sub_ = nh.subscribe("/ml_/pointcloud", 1, &KinematicChainVisualizer::lidar1_callback, this);
        lidar2_sub_ = nh.subscribe("/ml_/pointcloud2", 1, &KinematicChainVisualizer::lidar2_callback, this);
        pub_ = nh.advertise<visualization_msgs::MarkerArray>("/kinematic_chain_markers", 1);
        cloud1_pub_=nh.advertise<sensor_msgs::PointCloud2>("/lidar1_transformed", 1);
        cloud2_pub_=nh.advertise<sensor_msgs::PointCloud2>("/lidar2_transformed", 1);
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

        // Load lidar1_ext_t_
        if (!nh.getParam("lidar1_ext_t_", lidar1_ext_t_)) {
            ROS_WARN("Parameter 'lidar1_ext_t_' not found, using default: [0.0, 0.0, 0.0]");
            lidar1_ext_t_ = {0.0, 0.0, 0.0};
        } else if (lidar1_ext_t_.size() != 3) {
            ROS_ERROR("Parameter 'lidar1_ext_t_' must have exactly 3 elements, using default: [0.0, 0.0, 0.0]");
            lidar1_ext_t_ = {0.0, 0.0, 0.0};
        } else {
            ROS_INFO("Loaded lidar1_ext_t_: [%.6f, %.6f, %.6f]",
                     lidar1_ext_t_[0], lidar1_ext_t_[1], lidar1_ext_t_[2]);
        }

        // Load lidar1_ext_q_
        if (!nh.getParam("lidar1_ext_q_", lidar1_ext_q_)) {
            ROS_WARN("Parameter 'lidar1_ext_q_' not found, using default: [0.0, 0.0, 0.0, 1.0]");
            lidar1_ext_q_ = {0.0, 0.0, 0.0, 1.0};
        } else if (lidar1_ext_q_.size() != 4) {
            ROS_ERROR("Parameter 'lidar1_ext_q_' must have exactly 4 elements, using default: [0.0, 0.0, 0.0, 1.0]");
            lidar1_ext_q_ = {0.0, 0.0, 0.0, 1.0};
        } else {
            ROS_INFO("Loaded lidar1_ext_q_: [%.6f, %.6f, %.6f, %.6f]",
                     lidar1_ext_q_[0], lidar1_ext_q_[1], lidar1_ext_q_[2], lidar1_ext_q_[3]);
            // Verify and normalize quaternion
            Eigen::Quaternionf quat(lidar1_ext_q_[3], lidar1_ext_q_[0], lidar1_ext_q_[1], lidar1_ext_q_[2]);
            if (std::abs(quat.norm() - 1.0) > 1e-3) {
                ROS_WARN("lidar1_ext_q_ is not a unit quaternion (norm=%.6f), normalizing", quat.norm());
                quat.normalize();
                lidar1_ext_q_ = {quat.x(), quat.y(), quat.z(), quat.w()};
            }
        }

        prev_icp_trans=Eigen::Vector3f::Zero();

        lidar1_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        server_.reset(new interactive_markers::InteractiveMarkerServer("lidar_orientation_control"));
        createInteractiveMarker();

        nano_gicp_.setMaxCorrespondenceDistance(0.5);
        nano_gicp_.setNumThreads(20);
        nano_gicp_.setCorrespondenceRandomness(20);
        nano_gicp_.setMaximumIterations(10);
        nano_gicp_.setTransformationEpsilon(1e-6);
        nano_gicp_.setEuclideanFitnessEpsilon(1e-6);
        nano_gicp_.setRANSACIterations(50);
    }

private:
    ros::Subscriber sub_, lidar2_sub_, lidar1_sub_;
    ros::Publisher pub_, cloud1_pub_, cloud2_pub_;
    std::string frame_id_ = "world";
    std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
    double manual_rotate_roll_, manual_rotate_pitch_, manual_rotate_yaw_;
    double additional_rotate_roll_, additional_rotate_pitch_, additional_rotate_yaw_;
    tf2::Quaternion user_q_;
    bool use_marker_ = false;
    geometry_msgs::Point joint1_, joint2_;
    double manual_len_ratio_ = 0.5;
    //LiDAR1
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar1_cloud_;
    std::vector<double> lidar1_ext_t_, lidar1_ext_q_;
    bool joints_ready_ = false;
    double manual_x_add_=0.0,manual_y_add_=0.0,manual_z_add_=0.0;
    Eigen::Vector3f prev_icp_trans;
    std_msgs::Header lidar1_header_;
    Eigen::Matrix3f last_icp_rot_=Eigen::Matrix3f::Identity();
    Eigen::Vector3f last_icp_trans_=Eigen::Vector3f::Zero();
    bool icp_first_success_=false;
    double icp_threshold_;
    nano_gicp::NanoGICP<pcl::PointXYZINormal, pcl::PointXYZINormal> nano_gicp_;

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

    Eigen::Matrix4f make_DH_matrix(float theta, float d, float a, float alpha)
    {
        Eigen::Matrix4f mat;
        mat << cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta),
               sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta),
               0, sin(alpha), cos(alpha), d,
               0, 0, 0, 1;
        return mat;
    }

    void create_axis_markers(const Eigen::Matrix4f& transform, int id_offset, const std::string& frame_id,
                            ros::Time stamp, visualization_msgs::MarkerArray& marker_array,
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

    void DH_callback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        const std::vector<double>& data = msg->data;
        Eigen::Matrix4f boom_angle_DH_mat = make_DH_matrix(0,0.0, 0, -data[0]);
        Eigen::Matrix4f boom_len_DH_mat = make_DH_matrix(0, 5.7, 0, 0.0);
        Eigen::Matrix4f roll_mat;
        roll_mat << 1, 0, 0, 0,
                    0, cos(90.0*M_PI/180.0),-sin(90.0*M_PI/180.0),0,
                    0, sin(90.0*M_PI/180.0), cos(90.0*M_PI/180.0),0,
                    0,0,0,1;
        Eigen::Matrix4f origin = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f roll_transform = origin * roll_mat;
        Eigen::Matrix4f boom_yaw = roll_transform * boom_angle_DH_mat;
        Eigen::Matrix4f boom_pos = boom_yaw * boom_len_DH_mat;

        // Create MarkerArray
        visualization_msgs::MarkerArray marker_array;
        ros::Time stamp = ros::Time::now();

        // Origin frame (identity) - Red axes
        create_axis_markers(origin, 0, "world", stamp, marker_array, 1.0, 0.0, 0.0);
        // After roll (origin * roll_mat) - Green axes
        create_axis_markers(roll_transform, 3, "world", stamp, marker_array, 0.0, 1.0, 0.0);
        // Final boom position (origin * roll_mat * boom_DH_mat) - Blue axes
        create_axis_markers(boom_yaw, 6, "world", stamp, marker_array, 0.0, 0.0, 1.0);

        create_axis_markers(boom_pos, 9, "world", stamp, marker_array, 0.0, 0.0, 1.0);

        joint1_.x= 0.0;joint1_.y= 0.0;joint1_.z= 0.0;
        joint2_.x= boom_pos(0,3);joint2_.y= boom_pos(1,3);joint2_.z= boom_pos(2,3);

        joints_ready_ = true;
        // Publish
        pub_.publish(marker_array);
    }

    void lidar1_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZINormal>());
        lidar1_cloud_->points.clear();
        pcl::fromROSMsg(*msg, *cloud_in);

        // Convert lidar1_ext_q_ to Eigen::Matrix3f
        Eigen::Quaternionf quat(lidar1_ext_q_[3], lidar1_ext_q_[0], lidar1_ext_q_[1], lidar1_ext_q_[2]); // w, x, y, z
        quat.normalize();
        Eigen::Matrix3f rot = quat.toRotationMatrix();
        // ROS_INFO_STREAM("lidar1_rot_:\n" << rot);

        // Convert lidar1_ext_t_ to Eigen::Vector3f
        Eigen::Vector3f trans(lidar1_ext_t_[0], lidar1_ext_t_[1], lidar1_ext_t_[2]);
        
        for (const auto& pt : cloud_in->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rotated = rot * p;
            pcl::PointXYZINormal p_trans;
            p_trans.x = p_rotated.x() + trans.x();
            p_trans.y = p_rotated.y() + trans.y();
            p_trans.z = p_rotated.z() + trans.z();
            p_trans.intensity=0.0;
            lidar1_cloud_->points.push_back(p_trans);
        }

        sensor_msgs::PointCloud2 cloud_out;
        pcl::toROSMsg(*lidar1_cloud_, cloud_out);
        cloud_out.header.frame_id = "world";
        cloud_out.header.stamp = msg->header.stamp;
        cloud1_pub_.publish(cloud_out);
        lidar1_header_ = msg->header;
        return;
    }

    void lidar2_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        if(!joints_ready_) return;
        
        auto start_p1 = std::chrono::high_resolution_clock::now();
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::fromROSMsg(*msg, *cloud_in);
        auto end_p1 = std::chrono::high_resolution_clock::now();
        auto duration_p1 = std::chrono::duration_cast<std::chrono::microseconds>(end_p1 - start_p1).count();
        //ROS_INFO("1: ROS to PCL conversion took %ld microseconds", duration_p1);

        geometry_msgs::Point mid;
        double ratio = manual_len_ratio_;
        mid.x = ratio * joint1_.x + (1.0 - ratio) * joint2_.x+manual_x_add_;
        mid.y = ratio * joint1_.y + (1.0 - ratio) * joint2_.y+manual_y_add_;
        mid.z = ratio * joint1_.z + (1.0 - ratio) * joint2_.z+manual_z_add_;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZINormal>());
        auto end_p2 = std::chrono::high_resolution_clock::now();
        auto duration_p2 = std::chrono::duration_cast<std::chrono::microseconds>(end_p2 - end_p1).count();
        //ROS_INFO("2: Midpoint computation took %ld microseconds", duration_p2);

        // Build orthonormal basis
        Eigen::Vector3d x_axis(joint2_.x - joint1_.x,
                               joint2_.y - joint1_.y,
                               joint2_.z - joint1_.z);

        x_axis.normalize();

        Eigen::Vector3d temp_up(0.0, 0.0, 1.0);
        if (fabs(x_axis.dot(temp_up)) > 0.99) {
            temp_up = Eigen::Vector3d(0.0, 1.0, 0.0);
        }

        Eigen::Vector3d z_axis = x_axis.cross(temp_up).normalized();
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

        Eigen::Matrix3d rot_matrix;
        rot_matrix.col(0) = x_axis;
        rot_matrix.col(1) = y_axis;
        rot_matrix.col(2) = z_axis;

        tf2::Matrix3x3 tf_rot(
            rot_matrix(0, 0), rot_matrix(0, 1), rot_matrix(0, 2),
            rot_matrix(1, 0), rot_matrix(1, 1), rot_matrix(1, 2),
            rot_matrix(2, 0), rot_matrix(2, 1), rot_matrix(2, 2)
        );

        tf2::Quaternion base_q;
        tf_rot.getRotation(base_q);

        double roll_rad = manual_rotate_roll_ * M_PI / 180.0;
        double pitch_rad = manual_rotate_pitch_ * M_PI / 180.0;
        double yaw_rad = manual_rotate_yaw_ * M_PI / 180.0;

        tf2::Quaternion user_q;
        user_q.setRPY(roll_rad, pitch_rad, yaw_rad);

        tf2::Quaternion final_q = use_marker_ ? base_q * user_q_ : base_q * user_q;
        final_q.normalize();

        double roll, pitch, yaw;
        tf2::Matrix3x3(user_q_).getRPY(roll, pitch, yaw);
        roll = roll * 180.0 / M_PI;
        pitch = pitch * 180.0 / M_PI;
        yaw = yaw * 180.0 / M_PI;
        ROS_INFO("User RPY: Roll: %.2f, Pitch: %.2f, Yaw: %.2f", roll, pitch, yaw);

        tf2::Matrix3x3 rotation(final_q); // Use final_q_no_yaw

        Eigen::Matrix3f rot;
        rot << rotation[0][0], rotation[0][1], rotation[0][2],
              rotation[1][0], rotation[1][1], rotation[1][2],
              rotation[2][0], rotation[2][1], rotation[2][2];

        auto end_p3 = std::chrono::high_resolution_clock::now();
        auto duration_p3 = std::chrono::duration_cast<std::chrono::microseconds>(end_p3 - end_p2).count();
        //ROS_INFO("3: Orthonormal basis took %ld microseconds", duration_p3);  
        
        for (const auto& pt : cloud_in->points) 
        {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_rotated = rot * p;
            // Eigen::Vector3f p_rotated = p;
            pcl::PointXYZINormal p_trans;
            p_trans.x = p_rotated.x() + mid.x;
            p_trans.y = p_rotated.y() + mid.y;
            p_trans.z = p_rotated.z() + mid.z;
            p_trans.intensity=1.0;
            transformed_cloud->points.push_back(p_trans);
        }
        auto end_p4 = std::chrono::high_resolution_clock::now();
        auto duration_p4 = std::chrono::duration_cast<std::chrono::microseconds>(end_p4 - end_p3).count();
        //ROS_INFO("4: Transform cloud based on orthonormal took %ld microseconds", duration_p4);

        double add_roll_rad = additional_rotate_roll_ * M_PI / 180.0;
        double add_pitch_rad = additional_rotate_pitch_ * M_PI / 180.0;
        double add_yaw_rad = additional_rotate_yaw_ * M_PI / 180.0;
        tf2::Quaternion additional_q;
        additional_q.setRPY(add_roll_rad, add_pitch_rad, add_yaw_rad);

        tf2::Matrix3x3 additional_rotation(additional_q); // Use final_q_no_yaw

        Eigen::Matrix3f additional_rot;
        additional_rot << additional_rotation[0][0], additional_rotation[0][1], additional_rotation[0][2],
                          additional_rotation[1][0], additional_rotation[1][1], additional_rotation[1][2],
                          additional_rotation[2][0], additional_rotation[2][1], additional_rotation[2][2];

        double lidar1_2_diff_time = msg->header.stamp.toSec() - lidar1_header_.stamp.toSec();
        auto end_p5 = std::chrono::high_resolution_clock::now();
        auto duration_p5 = std::chrono::duration_cast<std::chrono::microseconds>(end_p5 - end_p4).count();
        //ROS_INFO("5: Additional rot took %ld microseconds", duration_p5);

        for (const auto& pt : transformed_cloud->points) 
        {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            if(p.norm()<3.0)
                continue;
            Eigen::Vector3f p_rotated = additional_rot * p;
            // Eigen::Vector3f p_rotated = p;
            pcl::PointXYZINormal p_trans;
            p_trans.x = p_rotated.x(); //+ mid.x;
            p_trans.y = p_rotated.y(); //+ mid.y;
            p_trans.z = p_rotated.z(); //+ mid.z;
            p_trans.intensity=1.0;
            p_trans.curvature=lidar1_2_diff_time;
            transformed_cloud2->points.push_back(p_trans);
        }
        auto end_p6 = std::chrono::high_resolution_clock::now();
        auto duration_p6 = std::chrono::duration_cast<std::chrono::microseconds>(end_p6 - end_p5).count();
        //ROS_INFO("6: Additional rot transform took %ld microseconds", duration_p6);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_aligned_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr gicp_aligned_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        double icp_score=-1.0;
        Eigen::Matrix3f icp_rot;
        Eigen::Vector3f icp_trans;
        if (lidar1_cloud_->points.size() > 0 && transformed_cloud2->points.size() > 0)
        {
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar1_cloud_voxelized(new pcl::PointCloud<pcl::PointXYZINormal>());
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud2_voxelized(new pcl::PointCloud<pcl::PointXYZINormal>());
        
            pcl::VoxelGrid<pcl::PointXYZINormal> voxel;
            voxel.setInputCloud(transformed_cloud2);
            voxel.setLeafSize(0.1f, 0.1f, 0.1f); // 1cm voxel size
            voxel.filter(*transformed_cloud2_voxelized);
            voxel.setInputCloud(lidar1_cloud_);
            voxel.filter(*lidar1_cloud_voxelized);

            auto end_p7 = std::chrono::high_resolution_clock::now();
            auto duration_p7 = std::chrono::duration_cast<std::chrono::microseconds>(end_p7 - end_p6).count();
            //ROS_INFO("7: Voxelization took %ld microseconds", duration_p7);

            
            pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
            icp.setInputSource(transformed_cloud2_voxelized); // Source
            icp.setInputTarget(lidar1_cloud_voxelized);     // Target

            // Set ICP parameters
            icp.setMaximumIterations(10);          // Max number of iterations
            icp.setTransformationEpsilon(1e-8);    // Convergence criterion for transformation
            icp.setMaxCorrespondenceDistance(0.5); // Max distance for point correspondences (adjust based on your data)
            icp.setEuclideanFitnessEpsilon(1e-8);  // Convergence criterion for fitness score
            auto end_p8 = std::chrono::high_resolution_clock::now();
            auto duration_p8 = std::chrono::duration_cast<std::chrono::microseconds>(end_p8 - end_p7).count();
            //ROS_INFO("8: Set point cloud took %ld microseconds", duration_p8);

            // Perform ICP
            icp.align(*icp_aligned_cloud);
            auto end_p9 = std::chrono::high_resolution_clock::now();
            auto duration_p9 = std::chrono::duration_cast<std::chrono::microseconds>(end_p9 - end_p8).count();
            //ROS_INFO("9: Align point cloud took %ld microseconds", duration_p9);

            // if (icp.hasConverged())
            // {
            //     ROS_INFO("ICP converged with score: %f", icp.getFitnessScore());
            //     Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
            //     ROS_INFO_STREAM("ICP Transformation Matrix:\n" << icp_transform);

            //     // Optionally, extract rotation and translation from icp_transform
            //     icp_rot = icp_transform.block<3, 3>(0, 0);
            //     icp_trans = icp_transform.block<3, 1>(0, 3);
            //     ROS_INFO("ICP Translation: [%.6f, %.6f, %.6f]", icp_trans.x(), icp_trans.y(), icp_trans.z());

            //     std::cout<<"(icp_trans-prev_icp_trans).norm():"<<(icp_trans-prev_icp_trans).norm()<<std::endl;
            //     if((icp_trans-prev_icp_trans).norm()<0.1)
            //     {
            //         icp_score=icp.getFitnessScore();
            //     }
                
            //     prev_icp_trans=icp_trans;

            //     // Convert rotation matrix to quaternion for logging
            //     Eigen::Quaternionf icp_quat(icp_rot);
            //     ROS_INFO("ICP Quaternion: [x: %.6f, y: %.6f, z: %.6f, w: %.6f]",
            //             icp_quat.x(), icp_quat.y(), icp_quat.z(), icp_quat.w());
            // }
            // else
            // {
            //     ROS_WARN("ICP did not converge");
            // }
            if(icp.hasConverged())
            {
                std::cout<<"ICP converge!"<<std::endl;
                nano_gicp_.setInputSource(icp_aligned_cloud); // Source
                nano_gicp_.calculateSourceCovariances();
                nano_gicp_.setInputTarget(lidar1_cloud_); // Target
                nano_gicp_.calculateTargetCovariances();
                auto end_p10 = std::chrono::high_resolution_clock::now();
                auto duration_p10 = std::chrono::duration_cast<std::chrono::microseconds>(end_p10 - end_p9).count();
                //ROS_INFO("10: GICP set took %ld microseconds", duration_p10);
                nano_gicp_.align(*icp_aligned_cloud);
                auto end_p11 = std::chrono::high_resolution_clock::now();
                auto duration_p11 = std::chrono::duration_cast<std::chrono::microseconds>(end_p11 - end_p10).count();
                //ROS_INFO("11: GICP align took %ld microseconds", duration_p11);
                ROS_INFO("GICP score: %f", nano_gicp_.getFitnessScore());

                if (nano_gicp_.getFitnessScore()<icp_threshold_)
                {
                    ROS_INFO("GICP converged with score: %f", nano_gicp_.getFitnessScore());
                    Eigen::Matrix4f icp_transform = nano_gicp_.getFinalTransformation();
                    ROS_INFO_STREAM("GICP Transformation Matrix:\n" << icp_transform);

                    // Optionally, extract rotation and translation from icp_transform
                    icp_rot = icp_transform.block<3, 3>(0, 0);
                    icp_trans = icp_transform.block<3, 1>(0, 3);
                    ROS_INFO("GICP Translation: [%.6f, %.6f, %.6f]", icp_trans.x(), icp_trans.y(), icp_trans.z());

                    std::cout<<"(gicp_trans-prev_icp_trans).norm():"<<(icp_trans-prev_icp_trans).norm()<<std::endl;
                    if((icp_trans-prev_icp_trans).norm()<0.1)
                    {
                        icp_score=nano_gicp_.getFitnessScore();
                    }
                    
                    prev_icp_trans=icp_trans;

                    // Convert rotation matrix to quaternion for logging
                    Eigen::Quaternionf icp_quat(icp_rot);
                    ROS_INFO("GICP Quaternion: [x: %.6f, y: %.6f, z: %.6f, w: %.6f]",
                            icp_quat.x(), icp_quat.y(), icp_quat.z(), icp_quat.w());
                    auto end_p12 = std::chrono::high_resolution_clock::now();
                    auto duration_p12 = std::chrono::duration_cast<std::chrono::microseconds>(end_p12 - end_p11).count();
                    //ROS_INFO("12: GICP transform took %ld microseconds", duration_p12);
                }
                else
                {
                    ROS_WARN("GICP did not converge");
                }
            }
            
        }
        else
        {
            ROS_WARN("ICP skipped: lidar1_cloud_ has %zu points, transformed_cloud2 has %zu points",
                    lidar1_cloud_->points.size(), transformed_cloud2->points.size());
            *icp_aligned_cloud = *transformed_cloud2; // Fallback to unaligned cloud
        }
        
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr publish_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        
        //              // lidar1_2_diff_time>0.0 && lidar1_2_diff_time<0.1)
        if( icp_score>0.0 && icp_score<icp_threshold_)
        {
            //*publish_cloud+=*icp_aligned_cloud; // Use ICP aligned cloud
            last_icp_rot_=icp_rot;
            last_icp_trans_=icp_trans;
            icp_first_success_=true;
        }
        // else if(icp_first_success_)
        // {
        //     std::cout << "Using last known ICP transform" << std::endl;
        //     // Modify transformed_cloud2 in-place
        //     for (auto& pt : transformed_cloud2->points) {
        //         Eigen::Vector3f p(pt.x, pt.y, pt.z);
        //         Eigen::Vector3f p_rotated = last_icp_rot_ * p + last_icp_trans_;
        //         pt.x = p_rotated.x();
        //         pt.y = p_rotated.y();
        //         pt.z = p_rotated.z();
        //         pt.intensity = 1.0;
        //         pt.curvature = lidar1_2_diff_time;
        //     }
        //     *publish_cloud += *transformed_cloud2;
        // }
        if (lidar1_cloud_->points.size() > 0) 
        {
            // Assuming transformed_cloud2 is pcl::PointCloud<pcl::PointXYZI>::Ptr
            for (auto& point : *transformed_cloud2) {
                point.intensity = 0.5; // Set a fixed value, e.g., 1.0
                // Or scale: point.intensity *= scaling_factor; // e.g., 0.5 for half intensity
                // Or offset: point.intensity += offset; // e.g., 10.0 to increase
            }
            //*publish_cloud += *transformed_cloud2;
            *publish_cloud += *lidar1_cloud_; // Combine point clouds
            ROS_INFO("Combined %zu points from lidar1_cloud_ with %zu points from transformed_cloud",
                     lidar1_cloud_->points.size(), transformed_cloud->points.size());
        }
        
        std::cout<<"lidar1-lidar2 time:"<<std::setprecision(17)<<lidar1_2_diff_time<<std::endl;
        
        // if(lidar1_2_diff_time>0.0 && lidar1_2_diff_time<0.1)
        {
            sensor_msgs::PointCloud2 cloud_out;
            pcl::toROSMsg(*publish_cloud, cloud_out);
            cloud_out.header.frame_id = "world";
            cloud_out.header.stamp = lidar1_header_.stamp; //+ ros::Duration(lidar1_2_diff_time);
            cloud2_pub_.publish(cloud_out);
        }
        auto end_p13 = std::chrono::high_resolution_clock::now();
        auto duration_p13 = std::chrono::duration_cast<std::chrono::microseconds>(end_p13 - end_p6).count();
        //ROS_INFO("13: Final publish took %ld microseconds", duration_p13);
        
        return;
    }


    // void lidar2_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    // {
    //     auto start_total = std::chrono::high_resolution_clock::now(); // Start total time

    //     if (!joints_ready_) return;

    //     // Process 1: Convert ROS message to PCL point cloud
    //     auto start_p1 = std::chrono::high_resolution_clock::now();
    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZINormal>());
    //     pcl::fromROSMsg(*msg, *cloud_in);
    //     auto end_p1 = std::chrono::high_resolution_clock::now();
    //     auto duration_p1 = std::chrono::duration_cast<std::chrono::microseconds>(end_p1 - start_p1).count();
    //     //ROS_INFO("1: ROS to PCL conversion took %ld microseconds", duration_p1);

    //     // Process 2: Compute midpoint
    //     auto start_p2 = std::chrono::high_resolution_clock::now();
    //     geometry_msgs::Point mid;
    //     double ratio = manual_len_ratio_;
    //     mid.x = ratio * joint1_.x + (1.0 - ratio) * joint2_.x + manual_x_add_;
    //     mid.y = ratio * joint1_.y + (1.0 - ratio) * joint2_.y + manual_y_add_;
    //     mid.z = ratio * joint1_.z + (1.0 - ratio) * joint2_.z + manual_z_add_;
    //     auto end_p2 = std::chrono::high_resolution_clock::now();
    //     auto duration_p2 = std::chrono::duration_cast<std::chrono::microseconds>(end_p2 - start_p2).count();
    //     //ROS_INFO("2: Midpoint computation took %ld microseconds", duration_p2);

    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZINormal>());

    //     // Process 3: Build orthonormal basis and rotation matrix
    //     auto start_p3 = std::chrono::high_resolution_clock::now();
    //     Eigen::Vector3d x_axis(joint2_.x - joint1_.x, joint2_.y - joint1_.y, joint2_.z - joint1_.z);
    //     x_axis.normalize();
    //     Eigen::Vector3d temp_up(0.0, 0.0, 1.0);
    //     if (fabs(x_axis.dot(temp_up)) > 0.99) {
    //         temp_up = Eigen::Vector3d(0.0, 1.0, 0.0);
    //     }
    //     Eigen::Vector3d z_axis = x_axis.cross(temp_up).normalized();
    //     Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    //     Eigen::Matrix3d rot_matrix;
    //     rot_matrix.col(0) = x_axis;
    //     rot_matrix.col(1) = y_axis;
    //     rot_matrix.col(2) = z_axis;
    //     tf2::Matrix3x3 tf_rot(
    //         rot_matrix(0, 0), rot_matrix(0, 1), rot_matrix(0, 2),
    //         rot_matrix(1, 0), rot_matrix(1, 1), rot_matrix(1, 2),
    //         rot_matrix(2, 0), rot_matrix(2, 1), rot_matrix(2, 2)
    //     );
    //     tf2::Quaternion base_q;
    //     tf_rot.getRotation(base_q);
    //     double roll_rad = manual_rotate_roll_ * M_PI / 180.0;
    //     double pitch_rad = manual_rotate_pitch_ * M_PI / 180.0;
    //     double yaw_rad = manual_rotate_yaw_ * M_PI / 180.0;
    //     tf2::Quaternion user_q;
    //     user_q.setRPY(roll_rad, pitch_rad, yaw_rad);
    //     tf2::Quaternion final_q = use_marker_ ? base_q * user_q_ : base_q * user_q;
    //     final_q.normalize();
    //     double roll, pitch, yaw;
    //     tf2::Matrix3x3(user_q_).getRPY(roll, pitch, yaw);
    //     roll = roll * 180.0 / M_PI;
    //     pitch = pitch * 180.0 / M_PI;
    //     yaw = yaw * 180.0 / M_PI;
    //     ROS_INFO("User RPY: Roll: %.2f, Pitch: %.2f, Yaw: %.2f", roll, pitch, yaw);
    //     tf2::Matrix3x3 rotation(final_q);
    //     Eigen::Matrix3f rot;
    //     rot << rotation[0][0], rotation[0][1], rotation[0][2],
    //         rotation[1][0], rotation[1][1], rotation[1][2],
    //         rotation[2][0], rotation[2][1], rotation[2][2];
    //     auto end_p3 = std::chrono::high_resolution_clock::now();
    //     auto duration_p3 = std::chrono::duration_cast<std::chrono::microseconds>(end_p3 - start_p3).count();
    //     //ROS_INFO("3: Orthonormal basis and rotation took %ld microseconds", duration_p3);

    //     // Process 4: First transformation (rotation and translation)
    //     auto start_p4 = std::chrono::high_resolution_clock::now();
    //     for (const auto& pt : cloud_in->points) 
    //     {
    //         Eigen::Vector3f p(pt.x, pt.y, pt.z);
    //         Eigen::Vector3f p_rotated = rot * p;
    //         pcl::PointXYZINormal p_trans;
    //         p_trans.x = p_rotated.x() + mid.x;
    //         p_trans.y = p_rotated.y() + mid.y;
    //         p_trans.z = p_rotated.z() + mid.z;
    //         p_trans.intensity = 1.0;
    //         transformed_cloud->points.push_back(p_trans);
    //     }
    //     auto end_p4 = std::chrono::high_resolution_clock::now();
    //     auto duration_p4 = std::chrono::duration_cast<std::chrono::microseconds>(end_p4 - start_p4).count();
    //     //ROS_INFO("4: First transformation took %ld microseconds", duration_p4);

    //     // Process 5: Additional rotation
    //     auto start_p5 = std::chrono::high_resolution_clock::now();
    //     double add_roll_rad = additional_rotate_roll_ * M_PI / 180.0;
    //     double add_pitch_rad = additional_rotate_pitch_ * M_PI / 180.0;
    //     double add_yaw_rad = additional_rotate_yaw_ * M_PI / 180.0;
    //     tf2::Quaternion additional_q;
    //     additional_q.setRPY(add_roll_rad, add_pitch_rad, add_yaw_rad);
    //     tf2::Matrix3x3 additional_rotation(additional_q);
    //     Eigen::Matrix3f additional_rot;
    //     additional_rot << additional_rotation[0][0], additional_rotation[0][1], additional_rotation[0][2],
    //                     additional_rotation[1][0], additional_rotation[1][1], additional_rotation[1][2],
    //                     additional_rotation[2][0], additional_rotation[2][1], additional_rotation[2][2];
    //     double lidar1_2_diff_time = msg->header.stamp.toSec() - lidar1_header_.stamp.toSec();
    //     for (const auto& pt : transformed_cloud->points) 
    //     {
    //         Eigen::Vector3f p(pt.x, pt.y, pt.z);
    //         if (p.norm() < 3.0)
    //             continue;
    //         Eigen::Vector3f p_rotated = additional_rot * p;
    //         pcl::PointXYZINormal p_trans;
    //         p_trans.x = p_rotated.x();
    //         p_trans.y = p_rotated.y();
    //         p_trans.z = p_rotated.z();
    //         p_trans.intensity = 1.0;
    //         p_trans.curvature = lidar1_2_diff_time;
    //         transformed_cloud2->points.push_back(p_trans);
    //     }
    //     auto end_p5 = std::chrono::high_resolution_clock::now();
    //     auto duration_p5 = std::chrono::duration_cast<std::chrono::microseconds>(end_p5 - start_p5).count();
    //     //ROS_INFO("5: Additional rotation took %ld microseconds", duration_p5);

    //     // Process 6: Voxel grid filtering
    //     auto start_p6 = std::chrono::high_resolution_clock::now();
    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr icp_aligned_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr gicp_aligned_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    //     double icp_score = -1.0;
    //     Eigen::Matrix3f icp_rot;
    //     Eigen::Vector3f icp_trans;
    //     if (lidar1_cloud_->points.size() > 0 && transformed_cloud2->points.size() > 0)
    //     {
    //         pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar1_cloud_voxelized(new pcl::PointCloud<pcl::PointXYZINormal>());
    //         pcl::VoxelGrid<pcl::PointXYZINormal> voxel;
    //         voxel.setInputCloud(transformed_cloud2);
    //         voxel.setLeafSize(0.1f, 0.1f, 0.1f);
    //         voxel.filter(*transformed_cloud2);
    //         voxel.setInputCloud(lidar1_cloud_);
    //         voxel.filter(*lidar1_cloud_voxelized);
    //     }
    //     auto end_p6 = std::chrono::high_resolution_clock::now();
    //     auto duration_p6 = std::chrono::duration_cast<std::chrono::microseconds>(end_p6 - start_p6).count();
    //     //ROS_INFO("6: Voxel grid filtering took %ld microseconds", duration_p6);

    //     // Process 7: ICP alignment
    //     auto start_p7 = std::chrono::high_resolution_clock::now();
    //     if (lidar1_cloud_->points.size() > 0 && transformed_cloud2->points.size() > 0)
    //     {
    //         pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar1_cloud_voxelized(new pcl::PointCloud<pcl::PointXYZINormal>());
    //         pcl::VoxelGrid<pcl::PointXYZINormal> voxel;
    //         voxel.setInputCloud(transformed_cloud2);
    //         voxel.setLeafSize(0.1f, 0.1f, 0.1f);
    //         voxel.filter(*transformed_cloud2);
    //         voxel.setInputCloud(lidar1_cloud_);
    //         voxel.filter(*lidar1_cloud_voxelized);

    //         pcl::IterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
    //         icp.setInputSource(transformed_cloud2);
    //         icp.setInputTarget(lidar1_cloud_voxelized);
    //         icp.setMaximumIterations(10);
    //         icp.setTransformationEpsilon(1e-8);
    //         icp.setMaxCorrespondenceDistance(0.5);
    //         icp.setEuclideanFitnessEpsilon(1e-8);
    //         icp.align(*icp_aligned_cloud);

    //         if (icp.hasConverged())
    //         {
    //             nano_gicp_.setInputSource(icp_aligned_cloud);
    //             nano_gicp_.calculateSourceCovariances();
    //             nano_gicp_.setInputTarget(lidar1_cloud_voxelized);
    //             nano_gicp_.calculateTargetCovariances();
    //             nano_gicp_.align(*icp_aligned_cloud);

    //             if (nano_gicp_.hasConverged())
    //             {
    //                 ROS_INFO("ICP converged with score: %f", nano_gicp_.getFitnessScore());
    //                 Eigen::Matrix4f icp_transform = nano_gicp_.getFinalTransformation();
    //                 ROS_INFO_STREAM("ICP Transformation Matrix:\n" << icp_transform);
    //                 icp_rot = icp_transform.block<3, 3>(0, 0);
    //                 icp_trans = icp_transform.block<3, 1>(0, 3);
    //                 ROS_INFO("ICP Translation: [%.6f, %.6f, %.6f]", icp_trans.x(), icp_trans.y(), icp_trans.z());
    //                 std::cout << "(icp_trans-prev_icp_trans).norm():" << (icp_trans - prev_icp_trans).norm() << std::endl;
    //                 if ((icp_trans - prev_icp_trans).norm() < 0.1)
    //                 {
    //                     icp_score = nano_gicp_.getFitnessScore();
    //                 }
    //                 prev_icp_trans = icp_trans;
    //                 Eigen::Quaternionf icp_quat(icp_rot);
    //                 ROS_INFO("ICP Quaternion: [x: %.6f, y: %.6f, z: %.6f, w: %.6f]",
    //                         icp_quat.x(), icp_quat.y(), icp_quat.z(), icp_quat.w());
    //             }
    //             else
    //             {
    //                 ROS_WARN("ICP did not converge");
    //             }
    //         }
    //     }
    //     else
    //     {
    //         ROS_WARN("ICP skipped: lidar1_cloud_ has %zu points, transformed_cloud2 has %zu points",
    //                 lidar1_cloud_->points.size(), transformed_cloud2->points.size());
    //         *icp_aligned_cloud = *transformed_cloud2;
    //     }
    //     auto end_p7 = std::chrono::high_resolution_clock::now();
    //     auto duration_p7 = std::chrono::duration_cast<std::chrono::microseconds>(end_p7 - start_p7).count();
    //     //ROS_INFO("7: ICP and GICP alignment took %ld microseconds", duration_p7);

    //     // Process 8: Publishing point cloud
    //     auto start_p8 = std::chrono::high_resolution_clock::now();
    //     pcl::PointCloud<pcl::PointXYZINormal>::Ptr publish_cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    //     if (icp_score > 0.0 && icp_score < icp_threshold_)
    //     {
    //         *publish_cloud += *icp_aligned_cloud;
    //         last_icp_rot_ = icp_rot;
    //         last_icp_trans_ = icp_trans;
    //         icp_first_success_ = true;
    //     }
    //     if (lidar1_cloud_->points.size() > 0) 
    //     {
    //         *publish_cloud += *lidar1_cloud_;
    //         ROS_INFO("Combined %zu points from lidar1_cloud_ with %zu points from transformed_cloud",
    //                 lidar1_cloud_->points.size(), transformed_cloud->points.size());
    //     }
    //     std::cout << "lidar1-lidar2 time:" << std::setprecision(17) << lidar1_2_diff_time << std::endl;
    //     {
    //         sensor_msgs::PointCloud2 cloud_out;
    //         pcl::toROSMsg(*publish_cloud, cloud_out);
    //         cloud_out.header.frame_id = "world";
    //         cloud_out.header.stamp = lidar1_header_.stamp;
    //         cloud2_pub_.publish(cloud_out);
    //     }
    //     auto end_p8 = std::chrono::high_resolution_clock::now();
    //     auto duration_p8 = std::chrono::duration_cast<std::chrono::microseconds>(end_p8 - start_p8).count();
    //     //ROS_INFO("8: Publishing point cloud took %ld microseconds", duration_p8);

    //     // Total execution time
    //     auto end_total = std::chrono::high_resolution_clock::now();
    //     auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    //     ROS_INFO("Total callback execution took %ld microseconds", duration_total);

    //     return;
    // }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinematic_chain");
    KinematicChainVisualizer visualizer;
    ros::spin();
    return 0;
}
