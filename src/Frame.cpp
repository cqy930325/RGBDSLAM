#include "Frame.hpp"

Frame::Frame(std::string rgb_path, std::string depth_path){//, cv::Ptr<cv::OrbFeatureDetector>& detector, cv::Ptr<cv::OrbDescriptorExtractor>& descriptor){
    read_rgb(rgb_path);
    read_depth(depth_path);
    //calc_keypoint(detector);
    //calc_desp(descriptor);
}

void Frame::read_rgb(std::string rgb_path){
    rgb = cv::imread(rgb_path);
}

void Frame::read_depth(std::string depth_path){
    depth = cv::imread(depth_path);
}

/*
 *void Frame::calc_keypoint(cv::Ptr<cv::OrbFeatureDetector>& detector){
 *    detector->detect(rgb, kp);
 *}
 *
 *void Frame::calc_desp(cv::Ptr<cv::OrbDescriptorExtractor>& descriptor){
 *    descriptor->compute(rgb, kp, desp);
 *}
 */


