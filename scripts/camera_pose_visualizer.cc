#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <fstream>

class PoseVisualizer {
public:
    PoseVisualizer() : logFile("/home/wang/catkin_ws/src/wla_orb/log/pose_log.txt") {
        if (!logFile.is_open()) {
            ROS_ERROR("Failed to open log file");
            return;
        }

        // Subscribe to the keyframe pose topic
        poseSub = nh.subscribe("/slam/keyframe_pose", 1, &PoseVisualizer::poseCallback, this);

        // Initialize the PCL visualizer
        viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0.0, 0.0, 0.0);
        viewer->initCameraParameters();
    }

    void run() {
        while (!viewer->wasStopped()) {
            viewer->spinOnce(500);
            ros::spinOnce();
            ros::Duration(0.1).sleep(); // Sleep for 100ms
        }
    }

private:
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    // Log pose to the file
    logFile << "Position: (" << msg->pose.position.x << ", " << msg->pose.position.y << ", " << msg->pose.position.z << ")"
            << " Orientation: (" << msg->pose.orientation.w << ", " << msg->pose.orientation.x << ", "
            << msg->pose.orientation.y << ", " << msg->pose.orientation.z << ")" << std::endl;

    // Convert ROS Pose to Eigen Transformation Matrix
    Eigen::Quaterniond q(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Matrix3d rotationMatrix = q.toRotationMatrix();
    Eigen::Vector3d translation(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    Eigen::Matrix4d poseMatrix = Eigen::Matrix4d::Identity();
    poseMatrix.block<3, 3>(0, 0) = rotationMatrix;
    poseMatrix.block<3, 1>(0, 3) = translation;

    // Invert Tcw to get Twc
    Eigen::Matrix4d Twc = poseMatrix.inverse(); // 求逆操作

    // Add the camera pose to the visualizer using Twc (world to camera transformation)
    addCameraToVisualizer(Twc);
    }

    void addCameraToVisualizer(const Eigen::Matrix4d& poseMatrix) {
        // Define the 4 corners of the camera rectangle in local camera coordinates
        std::vector<Eigen::Vector4d> cameraCorners = {
            {0.02, 0.01, 0.0, 1.0}, {0.02, -0.01, 0.0, 1.0}, {-0.02, -0.01, 0.0, 1.0}, {-0.02, 0.01, 0.0, 1.0}
        };

        // Transform the camera corners into the world coordinate frame
        std::vector<pcl::PointXYZ> worldCorners;
        for (const auto& corner : cameraCorners) {
            Eigen::Vector4d worldPoint = poseMatrix * corner;
            worldCorners.push_back(pcl::PointXYZ(worldPoint.x(), worldPoint.y(), worldPoint.z()));
        }

        // Draw the rectangle representing the camera
        static int cameraCounter = 0;
        std::string cameraId = "camera_" + std::to_string(cameraCounter);
        viewer->addLine(worldCorners[0], worldCorners[1], 1.0, 0.0, 0.0, cameraId + "_edge1");
        viewer->addLine(worldCorners[1], worldCorners[2], 1.0, 0.0, 0.0, cameraId + "_edge2");
        viewer->addLine(worldCorners[2], worldCorners[3], 1.0, 0.0, 0.0, cameraId + "_edge3");
        viewer->addLine(worldCorners[3], worldCorners[0], 1.0, 0.0, 0.0, cameraId + "_edge4");

        // Add the camera position to the path
        cameraPositions.push_back(pcl::PointXYZ(poseMatrix(0, 3), poseMatrix(1, 3), poseMatrix(2, 3)));
        if (cameraPositions.size() > 1) {
            viewer->addLine(cameraPositions[cameraPositions.size() - 2], cameraPositions.back(), 0.0, 1.0, 0.0, "path_" + std::to_string(cameraCounter));
        }

        cameraCounter++;
    }

    ros::NodeHandle nh;
    ros::Subscriber poseSub;
    std::ofstream logFile;
    pcl::visualization::PCLVisualizer::Ptr viewer;
    std::vector<pcl::PointXYZ> cameraPositions;  // Store camera positions to draw a path
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "PoseVisualizer");
    PoseVisualizer pv;
    pv.run();
    return 0;
}