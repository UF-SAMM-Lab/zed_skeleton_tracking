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

void zed_acquisition(int id, sl::Camera& zed, ros::Publisher joint_pub, ros::Publisher img_pub,ros::Publisher depth_pub,sl::Pose cam_pose, bool& run, sl::Timestamp& ts) {

    
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
    objectTracker_parameters_rt.detection_confidence_threshold = 40;

    std_msgs::Float32MultiArray joints_vec_msg;
    sensor_msgs::CompressedImage img_msg;

    sl::Objects bodies;  

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
            
            // Retrieve Detected Human Bodies
            zed.retrieveObjects(bodies, objectTracker_parameters_rt);
            joints_vec_msg.data.clear();
            // std::cout<<"camera "<<id<<" body count:"<<bodies.object_list.size()<<" at time "<<ts.getMilliseconds()-start_time<<std::endl;
            for(int o=0;o<std::min(1,int(bodies.object_list.size()));o++) {
                // std::cout<<"body:"<<o<<std::endl;
                for (auto& kp_3d:bodies.object_list[o].keypoint) {
                    // std::cout<<kp_3d<<std::endl;
                    for (int i=0;i<3;i++) {
                        joints_vec_msg.data.push_back(kp_3d[i]);
                    }
                    joints_vec_msg.data.push_back(1.0);
                }
                // std::cout<<joints_vec_msg<<std::endl;
            }
            if (!joints_vec_msg.data.empty()) joint_pub.publish(joints_vec_msg);
            zed.retrieveImage(image_left, sl::VIEW::LEFT, sl::MEM::CPU, display_resolution);  
            cv::Mat img_left_cv_mat(display_resolution.height,display_resolution.width, CV_8UC4,image_left.getPtr<sl::uchar1>(sl::MEM::CPU));
            render_2D(img_left_cv_mat, img_scale, bodies.object_list, true, sl::BODY_FORMAT::POSE_18);
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
            } else if (cam_info.serial_number==29191725) {
                pub_topic_name = "/cameraL/";
            } else {
                pub_topic_name = "/cameras/"+std::to_string(z);
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
            thread_pool[z] = std::thread(zed_acquisition, z,std::ref(zeds[z]), joint_pubs[z], img_pubs[z], depth_pubs[z],cam_poses[z], std::ref(run), std::ref(images_ts[z]));
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

