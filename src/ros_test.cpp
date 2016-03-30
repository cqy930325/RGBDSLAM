#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_bridge/kinect2_definitions.h>

class Receiver{
    boost::mutex lock;

    const std::string topicColor, topicDepth;
    bool updateImage;
    bool running;
    size_t frame;
    unsigned int queueSize;

    cv::Mat color, depth;
    cv::Mat cameraMatrixColor, cameraMatrixDepth;
    cv::Mat lookupX, lookupY;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

    ros::NodeHandle nh;
    ros::AsyncSpinner spinner;
    image_transport::ImageTransport it;
    image_transport::SubscriberFilter *subImageColor, *subImageDepth;
    message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;
    message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

    boost::thread image_viewer_thread;

    public:

    Receiver(const std::string &topicColor, const std::string &topicDepth)
        :topicColor(topicColor), topicDepth(topicDepth), nh("~"), spinner(0), it(nh), updateImage(false), running(false),frame(0){
            queueSize = 5;
            cm_rgb = cv::Mat::zeros(3, 3, CV_64F);  
            cm_depth = cv::Mat::zeros(3, 3, CV_64F);  
        }

    ~Receiver(){}

    void run(){
        start();
        stop();
    }

    private:

    void start(const Mode mode)
    {
        this->mode = mode;
        running = true;

        std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
        std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

        image_transport::TransportHints hints("compressed");
        subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
        subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
        subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
        subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

        syncApproximate = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncApproximate->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));

        spinner.start();
        while(!updateImage)
        {
            if(!ros::ok())
            {
                return;
            }
            boost::this_thread::sleep(boost::posix_time::milliseconds(1));
        }
    } 
    void stop()
    {
        spinner.stop();

        delete syncApproximate;

        delete subImageColor;
        delete subImageDepth;
        delete subCameraInfoColor;
        delete subCameraInfoDepth;

        running = false;
    }
    void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
            const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
    {
        cv::Mat color, depth;

        readCameraInfo(cameraInfoColor, cameraMatrixColor);
        readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
        readImage(imageColor, color);
        readImage(imageDepth, depth);

        // IR image input
        if(color.type() == CV_16U)
        {
            cv::Mat tmp;
            color.convertTo(tmp, CV_8U, 0.02);
            cv::cvtColor(tmp, color, CV_GRAY2BGR);
        }

        lock.lock();
        this->color = color;
        this->depth = depth;
        updateImage = true;
        lock.unlock();
    }

    void imageViewer()
    {
        cv::Mat color, depth, depthDisp, combined;
        double fps = 0;
        size_t frameCount = 0;
        std::ostringstream oss;
        const cv::Point pos(5, 15);
        const cv::Scalar colorText = CV_RGB(255, 255, 255);
        const double sizeText = 0.5;
        const int lineText = 1;
        const int font = cv::FONT_HERSHEY_SIMPLEX;

        cv::namedWindow("Image Viewer");
        oss << "starting...";

        for(; running && ros::ok();)
        {
            if(updateImage)
            {
                lock.lock();
                color = this->color;
                depth = this->depth;
                updateImage = false;
                lock.unlock();

                dispDepth(depth, depthDisp, 12000.0f);
                combine(color, depthDisp, combined);
                //combined = color;

                cv::putText(combined, oss.str(), pos, font, sizeText, colorText, lineText, CV_AA);
                cv::imshow("Image Viewer", combined);
            }

            int key = cv::waitKey(1);
            switch(key & 0xFF)
            {
                case 27:
                case 'q':
                    running = false;
                    break;
                case ' ':
                case 's':
                    if(mode == IMAGE)
                    {
                        saveImages(color, depth, depthDisp);
                    }
                    else
                    {
                        save = true;
                    }
                    break;
            }
        }
        cv::destroyAllWindows();
        cv::waitKey(100);
    } 
    void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
    {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
        pCvImage->image.copyTo(image);
    }

    void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
    {
        double *itC = cameraMatrix.ptr<double>(0, 0);
        for(size_t i = 0; i < 9; ++i, ++itC)
        {
            *itC = cameraInfo->K[i];
        }
    }

    void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
    {
        cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
        const uint32_t maxInt = 255;

#pragma omp parallel for
        for(int r = 0; r < in.rows; ++r)
        {
            const uint16_t *itI = in.ptr<uint16_t>(r);
            uint8_t *itO = tmp.ptr<uint8_t>(r);

            for(int c = 0; c < in.cols; ++c, ++itI, ++itO)
            {
                *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
            }
        }

        cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
    }

    void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
    {
        out = cv::Mat(inC.rows, inC.cols, CV_8UC3);

#pragma omp parallel for
        for(int r = 0; r < inC.rows; ++r)
        {
            const cv::Vec3b
                *itC = inC.ptr<cv::Vec3b>(r),
                *itD = inD.ptr<cv::Vec3b>(r);
            cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

            for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO)
            {
                itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
                itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
                itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
            }
        }
    }
    void saveImages(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
    {
        oss.str("");
        oss << "./" << std::setfill('0') << std::setw(4) << frame;
        const std::string baseName = oss.str();
        const std::string cloudName = baseName + "_cloud.pcd";
        const std::string colorName = baseName + "_color.jpg";
        const std::string depthName = baseName + "_depth.png";
        const std::string depthColoredName = baseName + "_depth_colored.png";

        OUT_INFO("saving cloud: " << cloudName);
        writer.writeBinary(cloudName, *cloud);
        OUT_INFO("saving color: " << colorName);
        cv::imwrite(colorName, color, params);
        OUT_INFO("saving depth: " << depthName);
        cv::imwrite(depthName, depth, params);
        OUT_INFO("saving depth: " << depthColoredName);
        cv::imwrite(depthColoredName, depthColored, params);
        OUT_INFO("saving complete!");
        ++frame;
    } 
};


int main(int argc, char **argv)
{
#if EXTENDED_OUTPUT
    ROSCONSOLE_AUTOINIT;
    if(!getenv("ROSCONSOLE_FORMAT"))
    {
        ros::console::g_formatter.tokens_.clear();
        ros::console::g_formatter.init("[${severity}] ${message}");
    }
#endif

    ros::init(argc, argv, "kinect2_viewer", ros::init_options::AnonymousName);

    if(!ros::ok())
    {
        return 0;
    }

    std::string ns = K2_DEFAULT_NS;
    std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
    std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    /*
     *"sd"
     *    topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
     *    topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
     */

    topicColor = "/" + ns + topicColor;
    topicDepth = "/" + ns + topicDepth;
    OUT_INFO("topic color: " FG_CYAN << topicColor << NO_COLOR);
    OUT_INFO("topic depth: " FG_CYAN << topicDepth << NO_COLOR);

    Receiver receiver(topicColor, topicDepth);

    OUT_INFO("starting receiver...");
    receiver.run();

    ros::shutdown();
    return 0;
}


