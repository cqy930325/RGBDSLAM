#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "constant.h"

typedef pcl::PointXYZRGBA point_t;
typedef pcl::PointCloud<point_t> point_cloud_t;

class PointCloud{
    public:
        PointCloud();
        ~PointCloud();
        void addFrame(frame_t &new_frame, tmat_t *T, bool init);
        void saveCloud(std::string path); 
        point_cloud_t::Ptr& getCloud();
    private:
        void image2cloud(frame_t &new_frame, point_cloud_t::Ptr &ret);
        point_cloud_t::Ptr pc;
};

#endif
