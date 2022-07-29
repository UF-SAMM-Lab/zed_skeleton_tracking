// ZED includes
#include <sl/Camera.hpp>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <thread>
#include <string>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <mutex>
#include <ros/package.h>

Eigen::MatrixXf transform_pixels_l(3,2);
Eigen::MatrixXf transform_pixels_r(3,2);
bool calibrate_transform = false;

void zed_acquisition(int id, sl::Camera& zed, ros::Publisher joint_pub, ros::Publisher img_pub,ros::Publisher depth_pub,sl::Pose cam_pose, bool& run, sl::Timestamp& ts, Eigen::MatrixXf& pixel, Eigen::Matrix3f& points3D, std::mutex& mtx) {

    
    sl::Plane floor_plane; // floor plane handle
    sl::Transform reset_from_floor_plane; // camera transform once floor plane is detected

    // Main Loop

    // Enable Positional tracking (mandatory for object detection)
    sl::PositionalTrackingParameters positional_tracking_parameters;
    bool need_floor_plane = positional_tracking_parameters.set_as_static;
    auto camera_config = zed.getCameraInformation().camera_configuration;

    // sl::Resolution pc_resolution(std::min((int)camera_config.resolution.width, 720), std::min((int)camera_config.resolution.height, 404));
	// // auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
	// sl::Mat point_cloud(pc_resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    // sl::Resolution display_resolution(std::min((int)camera_config.resolution.width, 1280), std::min((int)camera_config.resolution.height, 720));
    sl::Resolution display_resolution(camera_config.resolution.width, camera_config.resolution.height);


    // cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    // sl::Mat image_left(display_resolution, sl::MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::Mat image;
    sl::Mat point_cloud;
    // cv::Mat depth_img(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    sl::Mat depth_map(display_resolution,sl::MAT_TYPE::F32_C1);

    // Configure object detection runtime parameters
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = 40;

    std_msgs::Float32MultiArray joints_vec_msg;
    sensor_msgs::CompressedImage img_msg;
    sensor_msgs::Image depth_msg;

    sl::Objects bodies;

    double start_time = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getMilliseconds();

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
            for(int o=0;o<bodies.object_list.size();o++) {
                // std::cout<<"body:"<<o<<std::endl;
                for (auto& kp_3d:bodies.object_list[o].keypoint) {
                    // std::cout<<kp_3d<<std::endl;
                    for (int i=0;i<3;i++) {
                        joints_vec_msg.data.push_back(kp_3d[i]*0.001);
                    }
                    joints_vec_msg.data.push_back(1.0);
                }
                // std::cout<<joints_vec_msg<<std::endl;
            }
            if (!joints_vec_msg.data.empty()) joint_pub.publish(joints_vec_msg);
            zed.retrieveImage(image, sl::VIEW::LEFT);
            zed.retrieveMeasure(depth_map,sl::MEASURE::DEPTH);
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA); 
            mtx.lock();
            for (int i=0;i<pixel.rows();i++) {
                sl::float4 point3D;
                float depth_value=0;
                depth_map.getValue(pixel(i,0),pixel(i,1),&depth_value);
                // std::cout<<"cam:"<<id<<"px:"<<pixel.row(i)<<", depth:"<<depth_value<<std::endl;
                point_cloud.getValue(pixel(i,0),pixel(i,1),&point3D);
                points3D.col(i) = Eigen::Vector3f(point3D.x,point3D.y,point3D.z);
            }
            mtx.unlock();
            img_msg.format = "jpeg";
            cv::imencode(".jpg", cv::Mat(camera_config.resolution.height,camera_config.resolution.width, CV_8UC4,image.getPtr<sl::uchar1>(sl::MEM::CPU)), img_msg.data);
            img_pub.publish(img_msg);
            cv::Mat depth_img(camera_config.resolution.height,camera_config.resolution.width, CV_32FC1,depth_map.getPtr<sl::uchar1>(sl::MEM::CPU));
            depth_msg.height = depth_img.rows;
            depth_msg.width = depth_img.cols;
            depth_msg.encoding = "32FC1";
            int size = depth_img.total() * depth_img.elemSize();
            uint8_t *bytes = new uint8_t[size];  
            std::memcpy(bytes,depth_img.data,size * sizeof(uint8_t));
            depth_msg.data.clear();
            for (uint8_t *i=bytes;i!=bytes+size;++i) depth_msg.data.push_back(*i);
            delete[] bytes;

            depth_pub.publish(depth_msg);

            //put image_left into sensor_msgs/compressed image
            
			// zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
			// zed.getPosition(cam_pose, sl::REFERENCE_FRAME::WORLD);
            
            ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
        }
        sl::sleep_ms(1);
    }
    
    // Release objects
	// image_left.free();
	image.free();
    point_cloud.free();
    floor_plane.clear();
    bodies.object_list.clear();

    // Disable modules
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();
}

void calibrate_pts_callback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    Eigen::Vector3f px;
    for (int i=0;i<2;i++) {
        px[0] = msg->data[i*6];
        px[1] = msg->data[i*6+1];
        px[2] = msg->data[i*6+2];
        transform_pixels_l.col(i)= px;
        px[0] = msg->data[i*6+3];
        px[1] = msg->data[i*6+4];
        px[2] = msg->data[i*6+5];
        transform_pixels_r.col(i)= px;
    }
    calibrate_transform = true;
    ROS_INFO_STREAM("calibrating camera transforms");
}

int main(int argc, char **argv) {


    ros::init(argc,argv,"zed_tracking");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Subscriber sub_pts = nh.subscribe("/cam_robo_transform_pts",1,calibrate_pts_callback);

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

    std::vector<sl::Camera> zeds(nb_detected_zed); // Retrieve depthnb_detected_zed);
    std::vector<sl::Pose> cam_poses(nb_detected_zed);
    // std::vector<Objects> cam_bodies(nb_detected_zed);
    std::vector<ros::Publisher> joint_pubs(nb_detected_zed);
    std::vector<ros::Publisher> img_pubs(nb_detected_zed);
    std::vector<ros::Publisher> depth_pubs(nb_detected_zed);
    std::string pub_topic_name;
    int l_cam_id=0;
    int r_cam_id=0;
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
        init_parameters.depth_minimum_distance = 0.2;
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
                r_cam_id = z;
            } else if (cam_info.serial_number==29191725) {
                pub_topic_name = "/skeleton_L";
                l_cam_id = z;
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
    // std::vector<cv::Mat> images_lr(nb_detected_zed); // display images
    std::vector<sl::Timestamp> images_ts(nb_detected_zed); // images timestamps
    std::vector<sl::Timestamp> last_images_ts(nb_detected_zed); // images timestamps
    std::vector<Eigen::Matrix3f> px_points(nb_detected_zed); // images timestamps
    std::vector<Eigen::MatrixXf> pixels(nb_detected_zed); // images timestamps
    std::vector<std::mutex> mtx(nb_detected_zed); // images timestamps

    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) {
            // camera acquisition thread
            thread_pool[z] = std::thread(zed_acquisition, z,std::ref(zeds[z]), joint_pubs[z], img_pubs[z], depth_pubs[z],cam_poses[z], std::ref(run), std::ref(images_ts[z]),std::ref(pixels[z]),std::ref(px_points[z]),std::ref(mtx[z]));
        }
    bool calibrating = false;
    Eigen::Vector3f x_axis;
    Eigen::Vector3f y_axis;
    Eigen::Vector3f z_axis;
    Eigen::Matrix3f rotation_part;
    Eigen::Isometry3f transform_new_to_right_cam;
    Eigen::Isometry3f transform_new_to_left_cam;
    Eigen::Isometry3f transform_right_to_left_cam;
    Eigen::Isometry3f transform_fixed_to_right_cam;
    std::vector<Eigen::Isometry3f> transforms_right_to_left_cam;
    std::vector<Eigen::Isometry3f> transforms_fixed_to_right_cam;
    Eigen::Isometry3f transform_fixed_to_new = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f rz;
    // rz = Eigen::Quaternionf(0,0,0,1);
    // transform_fixed_to_new.linear() = Eigen::Matrix3f(Eigen::Quaternionf(0,0,0,1));
    transform_fixed_to_new.translation() = Eigen::Vector3f(-15,-17,0)*0.0254;
    while (ros::ok()) {

        if (calibrate_transform) {
            pixels[l_cam_id] = transform_pixels_l;
            pixels[r_cam_id] = transform_pixels_r;

            if ((images_ts[l_cam_id]>last_images_ts[l_cam_id]) && (images_ts[r_cam_id]>last_images_ts[r_cam_id])) {
                mtx[l_cam_id].lock();
                Eigen::Matrix3f px_points_l = px_points[l_cam_id];
                mtx[l_cam_id].unlock();
                mtx[r_cam_id].lock();
                Eigen::Matrix3f px_points_r = px_points[r_cam_id];
                mtx[r_cam_id].unlock();

                x_axis = px_points_l.col(1)-px_points_l.col(0);
                if (x_axis.norm()<0.05) continue;
                x_axis = x_axis.normalized();
                z_axis = x_axis.cross(px_points_l.col(2)-px_points_l.col(0));
                if (z_axis.norm()<0.05) continue;
                z_axis = z_axis.normalized();
                y_axis = z_axis.cross(x_axis);
                y_axis = y_axis.normalized();
                rotation_part.col(0) = x_axis;
                rotation_part.col(1) = y_axis;
                rotation_part.col(2) = z_axis;
                Eigen::Isometry3f transform_L = Eigen::Isometry3f::Identity();
                transform_L.linear() = rotation_part;
                transform_L.translation() = px_points_l.col(0);
                x_axis = px_points_r.col(1)-px_points_r.col(0);
                if (x_axis.norm()<0.05) continue;
                x_axis = x_axis.normalized();
                z_axis = x_axis.cross(px_points_r.col(2)-px_points_r.col(0));
                if (z_axis.norm()<0.05) continue;
                z_axis = z_axis.normalized();
                y_axis = z_axis.cross(x_axis);
                y_axis = y_axis.normalized();
                rotation_part.col(0) = x_axis;
                rotation_part.col(1) = y_axis;
                rotation_part.col(2) = z_axis;
                Eigen::Isometry3f transform_R = Eigen::Isometry3f::Identity();
                transform_R.linear() = rotation_part;
                transform_R.translation() = px_points_r.col(0);
                transform_new_to_right_cam = transform_R.inverse();
                transform_new_to_left_cam = transform_L.inverse();
                transform_right_to_left_cam = transform_R*transform_new_to_left_cam;
                std::cout<<"transform right to left cam\n:";
                std::cout<<transform_right_to_left_cam.matrix()<<std::endl;
                transforms_right_to_left_cam.push_back(transform_right_to_left_cam);
                transform_fixed_to_right_cam = transform_fixed_to_new*transform_new_to_right_cam;
                transforms_fixed_to_right_cam.push_back(transform_fixed_to_right_cam);
                std::cout<<"transform fixed to right cam\n:";
                std::cout<<transform_fixed_to_right_cam.matrix()<<std::endl;


                last_images_ts[l_cam_id] = images_ts[l_cam_id];
                last_images_ts[r_cam_id] = images_ts[r_cam_id];
                std::cout<<"summation size:"<<transforms_fixed_to_right_cam.size()<<std::endl;
                if (int(transforms_fixed_to_right_cam.size())>999) {
                    Eigen::Matrix4f transform_sums;
                    for (int i=0;i<1000;i++) {
                        transform_sums+=transforms_fixed_to_right_cam[i].matrix();
                        std::cout<<"max coeff:"<<transforms_fixed_to_right_cam[i].matrix().maxCoeff()<<std::endl;
                    }
                    transform_fixed_to_right_cam = transform_sums*(1.0/1000);
                    std::cout<<"mean transform fixed to right cam\n:";
                    std::cout<<transform_fixed_to_right_cam.matrix()<<std::endl;

                    std::string path = ros::package::getPath("zed_skeleton_tracking");
                    std::cout<<path<<std::endl;
                    std::cin.ignore();
                    // file pointer
                    std::ofstream myFile(path+"/src/T_fixed_c1.csv");
                
                    // opens an existing csv file or creates a new file.
                    Eigen::MatrixXf tf = transform_fixed_to_right_cam.matrix();
                    for (int i=0;i<4;i++) {
                        for (int j=0;j<4;j++) {
                            myFile<<tf(i,j);
                            if (j<4) myFile<<",";
                        }
                        if (i<4) myFile<<"\n";
                    }
                    myFile.close();
                    break;
                }
            }
            
        }


        ros::Duration(0.1).sleep();

    }

    // stop all running threads
    run = false;

    // Wait for every thread to be stopped
    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) 
            thread_pool[z].join();

    return EXIT_SUCCESS;

}

