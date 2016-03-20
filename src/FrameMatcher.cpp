#include "FrameMatcher.hpp"

FrameMatcher::FrameMatcher(){
    //TODO
    detector = new cv::OrbFeatureDetector(500, 1.0f, 2, 10, 0, 2, 0, 10);
    descriptor = new cv::OrbDescriptorExtractor();
    matcher = new cv::BFMatcher();
    /*
     *detector = new cv::SiftFeatureDetector();
     *descriptor = new cv::SiftDescriptorExtractor(); 
     *matcher = new cv::FlannBasedMatcher();
     */
}

FrameMatcher::~FrameMatcher(){
}

void FrameMatcher::processFrame(frame_t &new_frame){
    detector->detect(new_frame.rgb, new_frame.kp);
    descriptor->compute(new_frame.rgb, new_frame.kp, new_frame.desp);  
}

void FrameMatcher::convertToTMat(tmat_t* T, match_result_t *result){
    cv::Mat R;
    cv::Rodrigues(result->rvec, R);
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);
    *T = Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    //Eigen::Translation<double,3> trans(result->tvec.at<double>(0,0), result->tvec.at<double>(0,1), result->tvec.at<double>(0,2));
    *T = angle;
    (*T)(0,3) = result->tvec.at<double>(0,0); 
    (*T)(1,3) = result->tvec.at<double>(0,1); 
    (*T)(2,3) = result->tvec.at<double>(0,2);
}

void FrameMatcher::matchFrame(frame_t &new_frame, match_result_t *result, bool init){
    processFrame(new_frame);
    if(init){
        updateFrame(new_frame);
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
    if(good_matches.size() < 10){
        std::cout<<"too little good match: "<<good_matches.size()<<std::endl;
        result->inliers = -1;
        return;
    }
    std::vector<cv::Point3f> pts_obj;
    std::vector<cv::Point2f> pts_img;
    for (size_t i = 0; i < good_matches.size(); ++i) {
        cv::Point2f p = current_frame.kp[good_matches[i].queryIdx].pt;
        ushort d = current_frame.depth.ptr<ushort>(int(p.y))[int(p.x)];
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
    result->rvec = rvec;
    result->tvec = tvec;
    result->inliers = inliers.rows;
}

void FrameMatcher::updateFrame(frame_t &new_frame){
    current_frame.rgb = new_frame.rgb.clone();
    current_frame.depth = new_frame.depth.clone();
    current_frame.desp = new_frame.desp.clone();
    current_frame.kp = new_frame.kp;
}
