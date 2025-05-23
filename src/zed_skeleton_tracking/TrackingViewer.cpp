#include <zed_skeleton_tracking/TrackingViewer.hpp>

// -------------------------------------------------
//            2D LEFT VIEW
// -------------------------------------------------

#define VERBOSE_DISPLAY 0

template<typename T>
inline cv::Point2f cvt(T pt, sl::float2 scale) {
    return cv::Point2f(pt.x * scale.x, pt.y * scale.y);
}

template<typename T>
void createSKPrimitive(sl::BodyData& body, const std::vector<std::pair<T, T>>&map, sl::float2 img_scale, cv::Mat& left_display, cv::Rect &roi_render, cv::Scalar color) {

    for (const auto& parts : map) {
        auto kp_a = cvt(body.keypoint_2d[getIdx(parts.first)], img_scale);
        auto kp_b = cvt(body.keypoint_2d[getIdx(parts.second)], img_scale);
        if (roi_render.contains(kp_a) && roi_render.contains(kp_b)) {

#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
            cv::line(left_display, kp_a, kp_b, color, 1);
#else
            cv::line(left_display, kp_a, kp_b, color, 2, cv::LINE_AA);
#endif
        }
    }


    int i = 0;
    // skeleton joints
    for (auto& kp : body.keypoint_2d) {
        cv::Point2f cv_kp = cvt(kp, img_scale);
        if (roi_render.contains(cv_kp)) {
            cv::circle(left_display, cv_kp, 3, color, -1);
#if VERBOSE_DISPLAY
            //auto str = std::to_string(i++);
            auto str = std::string(sl::toString((sl::BODY_38_PARTS)i++));
            cv::putText(left_display, str, cv_kp, cv::FONT_HERSHEY_COMPLEX, 0.4/*font_size*/, cv::Scalar(255, 0, 0)/*font_Color*/, 1/*font_weight*/);
#endif
        }
    }
}



void render_2D(cv::Mat &left_display, sl::float2 img_scale, std::vector<sl::BodyData> &bodies, bool isTrackingON, bool fastRender) {
    cv::Mat overlay = left_display.clone();
    cv::Rect roi_render(0, 0, left_display.size().width, left_display.size().height);

    // render skeleton joints and bones
    for (auto &it : bodies)
        if (renderObject(it, isTrackingON)) {
            if (it.keypoint_2d.size()) {
                cv::Scalar color = generateColorID_u(it.id);

                if (fastRender) {
                    if (it.keypoint_2d.size() == 18)
                        createSKPrimitive(it, BODY_18_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 34)
                        createSKPrimitive(it, BODY_34_BONES, img_scale, left_display, roi_render, color);
                    // else if (it.keypoint_2d.size() > 34)
                    //     createSKPrimitive(it, BODY_BONES_FAST_RENDER, img_scale, left_display, roi_render, color);
                } else {
                    if (it.keypoint_2d.size() == 18)
                        createSKPrimitive(it, BODY_18_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 34)
                        createSKPrimitive(it, BODY_34_BONES, img_scale, left_display, roi_render, color);
                    else if (it.keypoint_2d.size() == 38)
                        createSKPrimitive(it, BODY_38_BONES, img_scale, left_display, roi_render, color);
                    else //if (it.keypoint_2d.size() == 70)
                        createSKPrimitive(it, BODY_70_BONES, img_scale, left_display, roi_render, color);
                }
            }
        }
    // Here, overlay is as the left image, but with opaque masks on each detected objects
    cv::addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
}


// #include <zed_skeleton_tracking/TrackingViewer.hpp>

// // -------------------------------------------------
// //            2D LEFT VIEW
// // -------------------------------------------------

// template<typename T>
// inline cv::Point2f cvt(T pt, sl::float2 scale) {
//     return cv::Point2f(pt.x * scale.x, pt.y * scale.y);
// }

// void render_2D(cv::Mat &left_display, sl::float2 img_scale, std::vector<sl::ObjectData> &objects, bool isTrackingON, sl::BODY_FORMAT body_format) {
//     cv::Mat overlay = left_display.clone();
//     cv::Rect roi_render(1, 1, left_display.size().width-2, left_display.size().height-2);

//     // render skeleton joints and bones
//     for (auto i = objects.rbegin(); i != objects.rend(); ++i) {
//         sl::ObjectData& obj = (*i);
//         if (renderObject(obj, isTrackingON)) {
//             if (obj.keypoint_2d.size()) {
//                 cv::Scalar color = generateColorID_u(obj.id);
// 				if (body_format == sl::BODY_FORMAT::BODY_18) {
// 					// skeleton bones
// 					for (const auto& parts : SKELETON_BONES) {
// 						auto kp_a = cvt(obj.keypoint_2d[getIdx(parts.first)], img_scale);
// 						auto kp_b = cvt(obj.keypoint_2d[getIdx(parts.second)], img_scale);
// 						if (roi_render.contains(kp_a) && roi_render.contains(kp_b))
// 						{

// #if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
// 							cv::line(left_display, kp_a, kp_b, color, 1);
// #else						
// 							cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);
// #endif
// 						}
// 					}
// 					auto hip_left = obj.keypoint_2d[getIdx(sl::BODY_18_PARTS::LEFT_HIP)];
// 					auto hip_right = obj.keypoint_2d[getIdx(sl::BODY_18_PARTS::RIGHT_HIP)];
// 					auto spine = (hip_left + hip_right) / 2;
// 					auto neck = obj.keypoint_2d[getIdx(sl::BODY_18_PARTS::NECK)];

// 					if (hip_left.x > 0 && hip_left.y > 0 && hip_right.x > 0 && hip_right.y > 0 && neck.x > 0 && neck.y > 0) {

// 						auto kp_a = cvt(spine, img_scale);
// 						auto kp_b = cvt(obj.keypoint_2d[getIdx(sl::BODY_18_PARTS::NECK)], img_scale);
// 						if (roi_render.contains(kp_a) && roi_render.contains(kp_b))
// 						{
// #if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
// 							cv::line(left_display, kp_a, kp_b, color, 1);
// #else
// 							cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);
// #endif
// 						}
// 					}

// 					// skeleton joints
// 					for (auto& kp : obj.keypoint_2d) {
// 						cv::Point2f cv_kp = cvt(kp, img_scale);
// 						if (roi_render.contains(cv_kp) && (!((cv_kp.x==0) && (cv_kp.y==0))))
// 							cv::circle(left_display, cv_kp, 3, color, -1);
// 					}
// 					cv::Point2f cv_kp = cvt(spine, img_scale);
// 					if (hip_left.x > 0 && hip_left.y > 0 && hip_right.x > 0 && hip_right.y > 0)
// 						cv::circle(left_display, cv_kp, 3, color, -1);
// 				}
// 				else if (body_format == sl::BODY_FORMAT::POSE_34) {
// 					// skeleton bones
// 					for (const auto& parts : sl::BODY_BONES_POSE_34) {
// 						auto kp_a = cvt(obj.keypoint_2d[getIdx(parts.first)], img_scale);
// 						auto kp_b = cvt(obj.keypoint_2d[getIdx(parts.second)], img_scale);
// 						if (roi_render.contains(kp_a) && roi_render.contains(kp_b))
// 						{

// #if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
// 							cv::line(left_display, kp_a, kp_b, color, 1);
// #else
// 							cv::line(left_display, kp_a, kp_b, color, 1, cv::LINE_AA);
// #endif
// 						}
// 					}

// 					// skeleton joints
// 					for (auto& kp : obj.keypoint_2d) {
// 						cv::Point2f cv_kp = cvt(kp, img_scale);
// 						if (roi_render.contains(cv_kp))
// 							cv::circle(left_display, cv_kp, 3, color, -1);
// 					}
// 				}
//             }

//         }
//     }
//     // Here, overlay is as the left image, but with opaque masks on each detected objects
//     cv::addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display);
// }
