#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZRGBA point_t;
typedef pcl::PointCloud<point_t> point_cloud_t;

class PointCloud{
    public:
        PointCloud();
        ~PointCloud();
        void addImage(cv::Mat& rgb, cv::Mat& depth);
        void saveCloud(std::string path); 
        point_cloud_t::Ptr& getCloud();
    private:
        point_cloud_t::Ptr pc;
};
