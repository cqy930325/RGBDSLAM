#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>


struct virtual_odo_t{
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
};

struct frame_t{
    cv::Mat rgb;
    cv::Mat depth;
    cv::Mat desp;
    std::vector<cv::KeyPoint> kp;
};


class FrameMatcher{
    public:
        FrameMatcher(); //Not Used
        ~FrameMatcher();
        void updateFrame(frame_t &new_frame, virtual_odo_t* vo, bool init); 
        cv::Ptr<cv::OrbFeatureDetector> detector;
        cv::Ptr<cv::OrbDescriptorExtractor> descriptor;
        cv::Ptr<cv::BFMatcher> matcher;
    private:
        void processFrame(frame_t &new_frame);
        frame_t current_frame;        
};
