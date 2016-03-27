#include "FrameMatcher.hpp"

FrameMatcher::FrameMatcher(){
    //TODO
    detector = cv::FeatureDetector::create("ORB");
    descriptor = cv::DescriptorExtractor::create("ORB");
    matcher = cv::DescriptorMatcher::create("BruteForce");
    /*
     *detector = new cv::OrbFeatureDetector(150, 1.0f, 2, 10, 0, 2, 0, 10);
     *descriptor = new cv::OrbDescriptorExtractor();
     *matcher = new cv::BFMatcher();
     */
}

FrameMatcher::FrameMatcher(std::string detector_name, std::string matcher_name){
    detector = cv::FeatureDetector::create(detector_name);
    descriptor = cv::DescriptorExtractor::create(detector_name);
    if(detector_name == "ORB" && matcher_name == "FlannBased"){
        matcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(20,10,2));
    }
    else{
        matcher = cv::DescriptorMatcher::create(matcher_name);
    }
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

void FrameMatcher::matchFrame(frame_t &f1, frame_t &f2, match_result_t *result, camera_param_t *camera){
    processFrame(f1);
    processFrame(f2);
    std::vector<cv::DMatch> matches;
    matcher->match(f1.desp, f2.desp, matches);
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
        cv::Point2f p = f1.kp[good_matches[i].queryIdx].pt;
        ushort d = f1.depth.ptr<ushort>(int(p.y))[int(p.x)];
        if(d == 0){
            continue;
        }
        pts_img.push_back(cv::Point2f(f2.kp[good_matches[i].trainIdx].pt));
        cv::Point3f pt(p.x, p.y, d);
        cv::Point3f pd;
        pd.z = double(pt.z) / camera->scale;
        pd.x = (pt.x - camera->cx)*pd.z / camera->fx;
        pd.y = (pt.y - camera->cy)*pd.z / camera->fy;
        pts_obj.push_back(pd);
    }
    if(pts_obj.size() < 5 || pts_img.size() < 5){
        std::cout<<"too little pts_obj size: "<<pts_obj.size()<<std::endl;
        result->inliers = -1;
        return;
    }
    double camera_matrix_data[3][3] = {
        {camera->fx, 0, camera->cx},
        {0, camera->fy, camera->cy},
        {0, 0, 1}
    };
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    result->rvec = rvec;
    result->tvec = tvec;
    result->inliers = inliers.rows;
    result->good_matches = good_matches.size();
    result->features = f2.kp.size();
}

