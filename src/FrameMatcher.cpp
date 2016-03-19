#include "FrameMatcher.hpp"
#include "constant.h"

FrameMatcher::FrameMatcher(){
    //TODO
    detector = new cv::OrbFeatureDetector(150, 1.0f, 2, 10, 0, 2, 0, 10);
    descriptor = new cv::OrbDescriptorExtractor();
    matcher = new cv::BFMatcher();
}

FrameMatcher::~FrameMatcher(){
}

void FrameMatcher::processFrame(frame_t &new_frame){
    detector->detect(new_frame.rgb, new_frame.kp);
    descriptor->compute(new_frame.rgb, new_frame.kp, new_frame.desp);  
}


void FrameMatcher::updateFrame(frame_t &new_frame, virtual_odo_t* vo, bool init){
    processFrame(new_frame);
    if(init){
        current_frame = new_frame;
        return;
    }
    std::vector<cv::DMatch> matches;
    matcher->match(current_frame.desp, new_frame.desp, matches);
    std::vector<cv::DMatch> good_matches;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].distance < min_dist){
            min_dist = matches[i].distance;
        }
    }
    min_dist = min_dist*4;
    for (size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].distance < min_dist){
            good_matches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point3f> pts_obj;
    std::vector<cv::Point2f> pts_img;

    for (size_t i = 0; i < good_matches.size(); ++i) {
        cv::Point2f p = current_frame.kp[good_matches[i].queryIdx].pt;
        ushort d = current_frame.depth.ptr(int(p.y))[int(p.x)];
        if(d == 0){
            continue;
        }
        pts_img.push_back(cv::Point2f(new_frame.kp[good_matches[i].trainIdx].pt));
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd;
        pd.z = double(pt.z) / camera_factor;
        pd.x = (pt.x - camera_cx)*pd.z / camera_fx;
        pd.y = (pt.y - camera_cy)*pd.z / camera_fy;
        pts_obj.push_back(pd);
    }
    double camera_matrix_data[3][3] = {
        {camera_fx, 0, camera_cx},
        {0, camera_fy, camera_cy},
        {0, 0, 1}
    };
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    vo->rvec = rvec;
    vo->tvec = tvec;
    vo->inliers = inliers;
}

