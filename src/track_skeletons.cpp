// ZED includes
#include <sl/Camera.hpp>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <thread>
#include <string>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <zed_skeleton_tracking/TrackingViewer.hpp>
#include <math.h>
#include <Eigen/Dense>

Eigen::Vector4d fit_depth_line(Eigen::VectorXd depths) {
    std::vector<int> good_pts;
    for (int i=0;i<depths.size();i++) good_pts.push_back(i);
    double slope = 0;
    double intercept = 0;
    while (true) {
        Eigen::ArrayXd x(depths.size());
        for (int i=0;i<depths.size();i++) x[i] = i;
        double x_bar = x.mean();
        slope = ((depths.array()-depths.mean())*(x-x_bar)).sum()/((x-x_bar)*(x-x_bar)).sum();
        intercept = depths[0]-slope*x[0];
        Eigen::ArrayXd diffs(depths.size());
        for (int i=0;i<depths.size();i++) {
            diffs[i] = depths[i]-slope*double(i)-intercept;
        }
        double std_dev = 2*std::sqrt((diffs - diffs.mean()).square().sum()/(diffs.size()-1));
        bool no_deviations = true;
        for (int i=0;i<diffs.size();i++) {
            if ((abs(diffs[i])>std_dev) || (isnan(depths[i]))) {
                Eigen::VectorXd new_depths(depths.size()-1);
                int k = 0;
                for (int j=0;j<depths.size();j++) {
                    if (j!=i) {
                        new_depths[k] = depths[j];
                        k++;
                    }
                }
                depths = new_depths;
                good_pts.erase(good_pts.begin()+i);
                no_deviations = false;
                break;
            }
        }
        if ((no_deviations) || (depths.size()==2)) break;
    }
    Eigen::Vector4d params;
    params << good_pts[0], slope, intercept, good_pts.size();
    return params;
}

void zed_acquisition(int id, sl::Camera& zed, ros::Publisher joint_pub, ros::Publisher img_pub,ros::Publisher depth_pub,sl::Pose cam_pose, bool& run, sl::Timestamp& ts, double dist_lim) {

    
    sl::Plane floor_plane; // floor plane handle
    sl::Transform reset_from_floor_plane; // camera transform once floor plane is detected

    // Main Loop

    // Enable Positional tracking (mandatory for object detection)
    sl::PositionalTrackingParameters positional_tracking_parameters;
    bool need_floor_plane = positional_tracking_parameters.set_as_static;
    auto camera_config = zed.getCameraInformation().camera_configuration;

    sl::Resolution pc_resolution(std::min((int)camera_config.resolution.width, 720), std::min((int)camera_config.resolution.height, 404));
	// auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
	// sl::Mat point_cloud(pc_resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    sl::Resolution display_resolution(std::min((int)camera_config.resolution.width, 640), std::min((int)camera_config.resolution.height, 360));
    sl::Mat image;

    // cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    sl::Mat image_left(display_resolution, sl::MAT_TYPE::U8_C4);//, image_left_ocv.data, image_left_ocv.step);

    // Configure object detection runtime parameters
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = 20;

    std_msgs::Float32MultiArray joints_vec_msg;
    sensor_msgs::CompressedImage img_msg;

    sl::Objects bodies;      
    sl::Mat depth_map(display_resolution,sl::MAT_TYPE::F32_C1);


    double start_time = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getMilliseconds();
    sl::float2 img_scale(display_resolution.width / (float)camera_config.resolution.width, display_resolution.height / (float) camera_config.resolution.height);
    
    while (run) {
        // grab current images and compute depth
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            if (need_floor_plane) {
                if (zed.findFloorPlane(floor_plane, reset_from_floor_plane) == sl::ERROR_CODE::SUCCESS) {
                    need_floor_plane = false;
                }
            }
            std::vector<sl::ObjectData> bods;
            // Retrieve Detected Human Bodies
            zed.retrieveObjects(bodies, objectTracker_parameters_rt);
            zed.retrieveMeasure(depth_map,sl::MEASURE::DEPTH);
            cv::Mat depth_img(camera_config.resolution.height,camera_config.resolution.width, CV_32FC1,depth_map.getPtr<float>(sl::MEM::CPU));

            joints_vec_msg.data.clear();
            // std::cout<<"camera "<<id<<" body count:"<<bodies.object_list.size()<<" at time "<<ts.getMilliseconds()-start_time<<std::endl;
            for(int o=0;o<int(bodies.object_list.size());o++) {
                // std::cout<<"body:"<<o<<std::endl;
                bool skip_bod = false;
                std::vector<float> data;
                std::vector<float> extra_data;
                for (int j=0;j<int(bodies.object_list[o].keypoint.size());j++) {
                // for (auto& kp_3d:bodies.object_list[o].keypoint) {
                    // std::cout<<kp_3d<<std::endl;
                    sl::float3 kp_3d = bodies.object_list[o].keypoint[j];
                    // std::cout<<o<<":"<<kp_3d[0]*kp_3d[0]+kp_3d[1]*kp_3d[1]+kp_3d[2]*kp_3d[2]<<std::endl;
                    if (kp_3d[0]*kp_3d[0]+kp_3d[1]*kp_3d[1]+kp_3d[2]*kp_3d[2]>dist_lim) {
                        skip_bod = true;
                        break;
                    }
                    // std::cout<<bodies.object_list[o].keypoint_confidence[j]<<std::endl;
                    for (int i=0;i<3;i++) {
                        data.push_back(kp_3d[i]);
                    }
                    float depth_changes = 1.0;
                    if (j==0) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[0];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[1];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            // Eigen::VectorXd depths(it.count);
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {   
                                float d = depth_img.at<float>(it.pos());
                                // depths[i] = d;
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of neck:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==1) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[1];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[8];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            Eigen::VectorXd depths(it.count);
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                depths[i] = d;
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                            Eigen::VectorXd ln1 = fit_depth_line(depths);
                            std::cout<<"line1:"<<depths.size()<<","<<ln1.transpose()<<std::endl;
                        }
                        p1 = bodies.object_list[o].keypoint_2d[1];
                        p2 = bodies.object_list[o].keypoint_2d[11];
                        it = cv::LineIterator(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        if (it.count>0) {
                            Eigen::VectorXd depths(it.count);
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                depths[i] = d;
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                            Eigen::VectorXd ln2 = fit_depth_line(depths);
                            std::cout<<"line2:"<<depths.size()<<","<<ln2.transpose()<<std::endl;
                            // for (int i=0;i<3;i++) extra_data.push_back(ln[i]);
                        }
                        // std::cout<<"confidence of spine:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==2) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[1];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[2];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of shoulder:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==3) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[3];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[2];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of elbow:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==4) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[3];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[4];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of wrist:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==5) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[5];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[1];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of shoulder:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==6) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[5];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[6];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of elbow:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if (j==7) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[6];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[7];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of wrist:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } else if ((j==8)||(j==11)) {
                        sl::float2 p1 = bodies.object_list[o].keypoint_2d[8];
                        sl::float2 p2 = bodies.object_list[o].keypoint_2d[11];
                        cv::LineIterator it(depth_img, cv::Point(int(p1[0]),int(p1[1])), cv::Point(int(p2[0]),int(p2[1])), 8);
                        depth_changes = 1.0;
                        if (it.count>0) {
                            depth_changes = 0;
                            float prev_depth = depth_img.at<float>(it.pos());
                            for(int i=0; i<it.count; i++)
                            {
                                float d = depth_img.at<float>(it.pos());
                                if (isfinite(d)) {
                                    if (!isfinite(prev_depth)) prev_depth = d;
                                    depth_changes+=abs(d-prev_depth);
                                    prev_depth = d;
                                }
                                it++;
                            }
                        }
                        // std::cout<<"confidence of hips:"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    } 
                    // else {
                    //     // std::cout<<"confidence of "<<j<<":"<<depth_changes<<","<<isfinite(depth_changes)<<std::endl;
                    // }
                    if (isnan(depth_changes)) depth_changes=1000;
                    data.push_back(depth_changes); //confidence
                }
                if (!skip_bod) {
                    bods.push_back(bodies.object_list[o]);
                    joints_vec_msg.data = data;
                }
                // std::cout<<joints_vec_msg<<std::endl;
            }
            if (!joints_vec_msg.data.empty()) joint_pub.publish(joints_vec_msg);
            zed.retrieveImage(image_left, sl::VIEW::LEFT, sl::MEM::CPU, display_resolution);  
            cv::Mat img_left_cv_mat(display_resolution.height,display_resolution.width, CV_8UC4,image_left.getPtr<sl::uchar1>(sl::MEM::CPU));
            render_2D(img_left_cv_mat, img_scale, bods, true, sl::BODY_FORMAT::POSE_18);
            // zed.retrieveImage(image, sl::VIEW::LEFT);            
            img_msg.format = "jpeg";
            cv::imencode(".jpg", img_left_cv_mat, img_msg.data);
            img_pub.publish(img_msg);
            //put image_left into sensor_msgs/compressed image
            
			// zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
			// zed.getPosition(cam_pose, sl::REFERENCE_FRAME::WORLD);
            
            ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        }
        sl::sleep_ms(1);
    }
    
    // Release objects
	image_left.free();
	// image.free();
    // point_cloud.free();
    floor_plane.clear();
    bodies.object_list.clear();

    // Disable modules
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();
}

int main(int argc, char **argv) {


    ros::init(argc,argv,"zed_tracking");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Create ZED objects
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    // On Jetson the object detection combined with an heavy depth mode could reduce the frame rate too much
    init_parameters.depth_mode = sl::DEPTH_MODE::QUALITY;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.sdk_verbose=1;

    // Enable Positional tracking (mandatory for object detection)
    sl::PositionalTrackingParameters positional_tracking_parameters;
    
    // Enable the Objects detection module
    sl::ObjectDetectionParameters obj_det_params;
    obj_det_params.enable_tracking = true; // track people across images flow
    obj_det_params.enable_body_fitting = false; // smooth skeletons moves
	obj_det_params.body_format = sl::BODY_FORMAT::POSE_18;
    obj_det_params.detection_model = sl::DETECTION_MODEL::HUMAN_BODY_ACCURATE;

    std::vector< sl::DeviceProperties> devList = sl::Camera::getDeviceList();

    int nb_detected_zed = int(devList.size());

	for (int z = 0; z < nb_detected_zed; z++) {
		std::cout << "ID : " << devList[z].id << " ,model : " << devList[z].camera_model << " , S/N : " << devList[z].serial_number << " , state : "<<devList[z].camera_state<<std::endl;
	}
	
    if (nb_detected_zed == 0) {
        std::cout << "No ZED Detected, exit program" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << nb_detected_zed << " ZED Detected" << std::endl;

    std::vector<sl::Camera> zeds(nb_detected_zed);
    std::vector<sl::Pose> cam_poses(nb_detected_zed);
    // std::vector<Objects> cam_bodies(nb_detected_zed);
    std::vector<ros::Publisher> joint_pubs(nb_detected_zed);
    std::vector<ros::Publisher> img_pubs(nb_detected_zed);
    std::vector<ros::Publisher> depth_pubs(nb_detected_zed);
    std::string pub_topic_name;
    std::vector<double> dist_lims(nb_detected_zed);
	// bool quit = false;

    // try to open every detected cameras
    for (int z = 0; z < nb_detected_zed; z++) {
        init_parameters.input.setFromCameraID(z);
        init_parameters.camera_resolution = sl::RESOLUTION::HD1080;
        init_parameters.camera_fps = 30;
        init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
        init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
        init_parameters.sdk_verbose=1;
        init_parameters.coordinate_units = sl::UNIT::METER;
        init_parameters.depth_minimum_distance = 0.5;
        sl::ERROR_CODE err = zeds[z].open(init_parameters);
        dist_lims[z] = 0;
        if (err == sl::ERROR_CODE::SUCCESS) {
            auto cam_info = zeds[z].getCameraInformation();
            std::cout << cam_info.camera_model << ", ID: " << z << ", SN: " << cam_info.serial_number << " Opened" << std::endl;
            
            err = zeds[z].enablePositionalTracking(positional_tracking_parameters);
            if (err != sl::ERROR_CODE::SUCCESS) {
                ROS_ERROR_STREAM("enable Positional Tracking"<< err<< "\nExit program.");
                zeds[z].close();
                continue;
            }
            err = zeds[z].enableObjectDetection(obj_det_params);
            if (err != sl::ERROR_CODE::SUCCESS) {
                ROS_ERROR_STREAM("enable Object Detection"<< err<< "\nExit program.");
                zeds[z].close();
                continue;
            }
            
	        cam_poses[z].pose_data.setIdentity();
            if (cam_info.serial_number==29580072) {
                pub_topic_name = "/skeleton_R";
            } else if (cam_info.serial_number==29191725) {
                pub_topic_name = "/skeleton_L";
            } else {
                pub_topic_name = "/skeletons/"+std::to_string(z);
            }
            joint_pubs[z] = nh.advertise<std_msgs::Float32MultiArray>(pub_topic_name,0);
            if (cam_info.serial_number==29580072) {
                pub_topic_name = "/cameraR/";
                dist_lims[z] = 9;
            } else if (cam_info.serial_number==29191725) {
                pub_topic_name = "/cameraL/";
                dist_lims[z] = 16;
            } else {
                pub_topic_name = "/cameras/"+std::to_string(z);
                dist_lims[z] = 16;
            }
            img_pubs[z] = nh.advertise<sensor_msgs::CompressedImage>(pub_topic_name+"image_raw/compressed",0);
            depth_pubs[z] = nh.advertise<sensor_msgs::Image>(pub_topic_name+"depth_image",0);
        } else {
            std::cout << "ZED ID:" << z << " Error: " << err << std::endl;
            zeds[z].close();
        }
    }
    
    bool run = true;
    // Create a grab thread for each opened camera
    std::vector<std::thread> thread_pool(nb_detected_zed); // compute threads
    std::vector<cv::Mat> images_lr(nb_detected_zed); // display images
    std::vector<sl::Timestamp> images_ts(nb_detected_zed); // images timestamps

    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) {
            // create an image to store Left+Depth image
            images_lr[z] = cv::Mat(404, 720*2, CV_8UC4);
            // camera acquisition thread
            thread_pool[z] = std::thread(zed_acquisition, z,std::ref(zeds[z]), joint_pubs[z], img_pubs[z], depth_pubs[z],cam_poses[z], std::ref(run), std::ref(images_ts[z]),dist_lims[z]);
        }

    ros::waitForShutdown();

    // stop all running threads
    run = false;

    // Wait for every thread to be stopped
    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) 
            thread_pool[z].join();

    return EXIT_SUCCESS;

}

