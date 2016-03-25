#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "PointCloud.hpp"
#include "FrameMatcher.hpp"

struct config_t{
    std::string rgb_path;
    std::string depth_path;
    int min_inliers;
    double max_norm;
    int min_good_match;
    int read_count;
    int end_count;
    int read_bw;
    std::string detector_name;
    std::string matcher_name;
};

struct state_t{
    PointCloud cloud;
    FrameMatcher *fm;
    boost::mutex cloud_mutex;
    float time_count;
    int success_process;
    match_result_t r;
};

void match(state_t* state, config_t* config, camera_param_t *camera){
    while(1){
        state->cloud_mutex.lock();
        if(config->read_count > config->end_count){
            state->cloud_mutex.unlock();
            break;
        }
        std::cout<<config->read_count<<std::endl;
        frame_t f;
        if(config->read_bw){
            f.rgb = cv::imread(config->rgb_path+boost::lexical_cast<std::string>(config->read_count)+".png", CV_LOAD_IMAGE_GRAYSCALE);
        }
        else{
            f.rgb = cv::imread(config->rgb_path+boost::lexical_cast<std::string>(config->read_count)+".png");
        }
        f.depth = cv::imread(config->depth_path+boost::lexical_cast<std::string>(config->read_count)+".png", -1);
        match_result_t result;
        state->fm->matchFrame(f, &result, camera, false);
        if(result.inliers < config->min_inliers){
            std::cout<<"inlier skip"<<std::endl;
            config->read_count++;
            state->cloud_mutex.unlock();
            continue;
        }
        double norm = fabs(std::min(cv::norm(result.rvec), 2*M_PI-cv::norm(result.rvec)))+ fabs(cv::norm(result.tvec));  
        //std::cout<<"norm: "<<norm<<std::endl;
        if(norm >= config->max_norm){
            std::cout<<"norm skip"<<std::endl;
            config->read_count++;
            state->cloud_mutex.unlock();
            continue;
        }
        tmat_t trans;
        state->fm->convertToTMat(&trans, &result);
        state->success_process += 1;
        state->r.inliers += result.inliers;
        state->r.good_matches += result.good_matches;
        state->r.features += result.features;
        std::cout<<"average process time:"<<state->time_count/state->success_process<<std::endl;
        std::cout<<"average process features:"<<((float)state->r.features)/state->success_process<<std::endl;
        std::cout<<"average process inliers:"<<((float)state->r.inliers)/state->success_process<<std::endl;
        std::cout<<"average process good_matches:"<<((float)state->r.good_matches)/state->success_process<<std::endl;
        //std::cout<<"T = "<<trans.matrix()<<std::endl;
        state->cloud.addFrame(f, &trans, camera, false);
        state->fm->updateFrame(f);
        config->read_count++;
        //std::cout<<"leave"<<std::endl;
        state->cloud_mutex.unlock();
    }
}


int main(int argc, char *argv[])
{
    config_t config;
    boost::property_tree::ptree pt;
    boost::property_tree::read_json("./config.json", pt);
    config.read_count = pt.get<int>("start_idx");
    config.end_count = pt.get<int>("end_idx");
    config.rgb_path = pt.get<std::string>("rgb_path");
    config.depth_path = pt.get<std::string>("depth_path");
    config.min_good_match = pt.get<int>("min_good_match");
    config.min_inliers = pt.get<int>("min_inliers");
    config.max_norm = pt.get<double>("max_norm");
    config.read_bw = pt.get<int>("read_bw");
    config.detector_name = pt.get<std::string>("detector_name");
    config.matcher_name = pt.get<std::string>("matcher_name");
   
    camera_param_t camera;
    boost::property_tree::ptree cam = pt.get_child("camera_matrix"); 
    camera.fx = cam.get<double>("fx");
    camera.fy = cam.get<double>("fy");
    camera.cx = cam.get<double>("cx");
    camera.cy = cam.get<double>("cy");
    camera.scale = cam.get<double>("scale");

    state_t state;
    state.time_count = 0.0;
    state.success_process = 0;
    state.r.inliers = 0;
    state.r.good_matches = 0;
    state.r.features = 0;
    state.fm = new FrameMatcher(config.detector_name, config.matcher_name);

    frame_t f1;
    if(config.read_bw){
        f1.rgb = cv::imread(config.rgb_path+boost::lexical_cast<std::string>(config.read_count)+".png", CV_LOAD_IMAGE_GRAYSCALE);
    }
    else{
        f1.rgb = cv::imread(config.rgb_path+boost::lexical_cast<std::string>(config.read_count)+".png");
    }
    f1.depth = cv::imread(config.depth_path+boost::lexical_cast<std::string>(config.read_count)+".png", -1);
    config.read_count++;
    state.fm->matchFrame(f1, NULL, &camera, true);
    state.cloud.addFrame(f1, NULL, &camera, true);
    
    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud(state.cloud.getCloud(), "rgbd");

    boost::thread matchThread(match, &state, &config, &camera);
    matchThread.detach();

    while(1){
        state.cloud_mutex.lock();
        viz.updatePointCloud(state.cloud.getCloud(), "rgbd");
        viz.spinOnce(100);
        state.cloud_mutex.unlock();
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    delete state.fm;
    return 0;
}
