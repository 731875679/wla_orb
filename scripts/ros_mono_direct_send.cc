#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <sys/stat.h> // For mkdir function
#include <sys/types.h>
#include <errno.h>    // For errno
#include <string.h>   // For strerror
#include <geometry_msgs/PoseStamped.h>  // 用于发布相机位姿
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include "../include/System.h"

using namespace std;

class ImageGrabber {
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ros::Publisher& pose_pub, ros::Publisher& rgb_pub, ros::Publisher& points_pub)
        : mpSLAM(pSLAM), posePublisher(pose_pub), rgbPublisher(rgb_pub), pointPublisher(points_pub) {
        prevKeyFrameId = -1;
    }
    
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    
    geometry_msgs::PoseStamped CreatePoseMessage(ORB_SLAM3::KeyFrame* pKF, bool finishedLoop);
    void storeRGBImage(const sensor_msgs::ImageConstPtr& msg, int keyframe_id);
    sensor_msgs::PointCloud2 createMapPoints(ORB_SLAM3::KeyFrame* pKF);

private:
    ORB_SLAM3::System* mpSLAM;
    ros::Publisher posePublisher;
    ros::Publisher rgbPublisher;
    ros::Publisher pointPublisher;
    sensor_msgs::ImagePtr lastRGBImage;
    int prevKeyFrameId;

    std::unordered_set<size_t> publishedPointIDs;  // Set to store the IDs of already published MapPoints
};

// Define the method to create PoseStamped messages
geometry_msgs::PoseStamped ImageGrabber::CreatePoseMessage(ORB_SLAM3::KeyFrame* pKF, bool finishedLoop) {
    Sophus::SE3f Tcw = pKF->GetPose();   
    
    Eigen::Quaternionf q = Tcw.unit_quaternion(); 
    Eigen::Vector3f t = Tcw.translation(); 

    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.stamp = ros::Time(pKF->mTimeStamp);  
    poseMsg.header.frame_id = std::to_string(pKF->mnId) + "_" + (finishedLoop ? "1" : "0");

    poseMsg.pose.position.x = t(0);
    poseMsg.pose.position.y = t(1);
    poseMsg.pose.position.z = t(2);
    poseMsg.pose.orientation.w = q.w();
    poseMsg.pose.orientation.x = q.x();
    poseMsg.pose.orientation.y = q.y();
    poseMsg.pose.orientation.z = q.z();

    return poseMsg;
}

// Store RGB image temporarily
void ImageGrabber::storeRGBImage(const sensor_msgs::ImageConstPtr& msg, int keyframe_id) {
    lastRGBImage = sensor_msgs::ImagePtr(new sensor_msgs::Image(*msg));  // 深拷贝图像消息
    lastRGBImage->header.frame_id = std::to_string(keyframe_id);  // 将关键帧ID加入到frame_id
}

// Modify the createMapPoints function
sensor_msgs::PointCloud2 ImageGrabber::createMapPoints(ORB_SLAM3::KeyFrame* pKF) {
    const std::vector<ORB_SLAM3::MapPoint*>& vpMapPoints = pKF->GetMapPointMatches();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->header.frame_id = std::to_string(pKF->mnId);
    cloud->header.stamp = pcl_conversions::toPCL(ros::Time::now());

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(lastRGBImage, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return sensor_msgs::PointCloud2();
    }

    cv::Mat image = cv_ptr->image;

    for (size_t i = 0; i < vpMapPoints.size(); i++) {
        ORB_SLAM3::MapPoint* pMP = vpMapPoints[i];
        if (pMP && !pMP->isBad()) {
            size_t pointID = pMP->mnId;

            // Check if this MapPoint has already been published
            if (publishedPointIDs.find(pointID) == publishedPointIDs.end()) {
                // Get the 3D coordinates of the point
                Eigen::Vector3f pos = pMP->GetWorldPos();
                pcl::PointXYZRGB point;
                point.x = pos.x();
                point.y = pos.y();
                point.z = pos.z();

                // Get RGB from the current image
                int u = pKF->mvKeysUn[i].pt.x;
                int v = pKF->mvKeysUn[i].pt.y;

                if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                    cv::Vec3b rgb = image.at<cv::Vec3b>(v, u);
                    point.r = rgb[2];
                    point.g = rgb[1];
                    point.b = rgb[0];
                }
                
                cloud->points.push_back(point);
                publishedPointIDs.insert(pointID);
            }
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);

    return output;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg) {

    // Convert the ROS image message to cv::Mat
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Track the image using the SLAM system
    mpSLAM->TrackMonocular(cv_ptr->image, cv_ptr->header.stamp.toSec());

    // Check if loop closing is complete or new keyframe is available
    bool finishedLoop = mpSLAM->mpLocalMapper->finishedLoop;
    bool haveNewKeyframe = mpSLAM->mpTracker->newKeyFrame;

    if (finishedLoop || haveNewKeyframe) {
        // If loop closure is complete, gather all keyframe poses
        if (finishedLoop) {
            std::vector<ORB_SLAM3::KeyFrame*> allKeyFrames = mpSLAM->GetAtlas()->GetAllKeyFrames();
            for (ORB_SLAM3::KeyFrame* pKF : allKeyFrames) {
                if (pKF->mnId > prevKeyFrameId) {
                    geometry_msgs::PoseStamped poseMsg = CreatePoseMessage(pKF, true);
                    posePublisher.publish(poseMsg);
                    prevKeyFrameId = pKF->mnId;
                }
            }
        } else {
            // If not loop closure, only process the new keyframe
            ORB_SLAM3::KeyFrame* pKF = mpSLAM->mpTracker->GetLastKeyFrame();
            if (pKF && pKF->mnId != prevKeyFrameId) {
                storeRGBImage(msg, pKF->mnId);

                geometry_msgs::PoseStamped poseMsg = CreatePoseMessage(pKF, false);
                posePublisher.publish(poseMsg);

                rgbPublisher.publish(lastRGBImage);

                sensor_msgs::PointCloud2 mapPointsMsg = createMapPoints(pKF);
                pointPublisher.publish(mapPointsMsg);

                prevKeyFrameId = pKF->mnId;
            }
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "Mono");
    ros::start();

    if (argc != 3) {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // Initialize SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);

    ros::NodeHandle nodeHandler;

    // Publisher for keyframe pose
    ros::Publisher pose_pub = nodeHandler.advertise<geometry_msgs::PoseStamped>("/slam/keyframe_pose", 100);

    // Publisher for RGB image
    ros::Publisher image_pub = nodeHandler.advertise<sensor_msgs::Image>("/slam/keyframe_image", 100);
    ros::Publisher point3d_pub = nodeHandler.advertise<sensor_msgs::PointCloud2>("/slam/keyframe_point3d", 100);

    // Create ImageGrabber and subscribe to image topic
    ImageGrabber igb(&SLAM, pose_pub, image_pub, point3d_pub);
    ros::Subscriber sub = nodeHandler.subscribe("/camera/image_raw", 100, &ImageGrabber::GrabImage, &igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    ros::shutdown();

    return 0;
}