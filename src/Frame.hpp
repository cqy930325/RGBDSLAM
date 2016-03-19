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

class Frame{
    public:
        Frame(){}
        Frame(std::string rgb_path, std::string depth_path);//, cv::Ptr<cv::OrbFeatureDetector>& detector, cv::Ptr<cv::OrbDescriptorExtractor>& descriptor);
        ~Frame();
        void read_rgb(std::string rgb_path);
        void read_depth(std::string depth_path);
        //void calc_keypoint(cv::Ptr<cv::OrbFeatureDetector>& detector);
        //void calc_desp(cv::Ptr<cv::OrbDescriptorExtractor>& descriptor);
        cv::Mat rgb;
        cv::Mat depth;
        cv::Mat desp;
        std::vector<cv::KeyPoint> kp;
};


