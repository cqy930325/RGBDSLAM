#ifndef FRAME_MATCHER_HPP
#define FRAME_MATCHER_HPP

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constant.h"

class FrameMatcher{
    public:
        FrameMatcher();
        FrameMatcher(std::string detector_name, std::string matcher_name);
        ~FrameMatcher();
        void matchFrame(frame_t &new_frame, match_result_t *result, camera_param_t *camera, bool init); 
        void convertToTMat(tmat_t* T, match_result_t *result);
        void updateFrame(frame_t &new_frame);
        /*
         *cv::Ptr<cv::OrbFeatureDetector> detector;
         *cv::Ptr<cv::OrbDescriptorExtractor> descriptor;
         *cv::Ptr<cv::BFMatcher> matcher;
         */
        /*
         *cv::Ptr<cv::SiftFeatureDetector> detector;
         *cv::Ptr<cv::SiftDescriptorExtractor> descriptor;
         *cv::Ptr<cv::FlannBasedMatcher> matcher;
         */
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        //cv::Ptr<cv::FlannBasedMatcher> matcher;
    private:
        void processFrame(frame_t &new_frame);
        frame_t current_frame;        
};

#endif
