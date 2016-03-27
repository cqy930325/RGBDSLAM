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

//g2o header file
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include "PointCloud.hpp"
#include "FrameMatcher.hpp"

typedef g2o::BlockSolver_6_3 BlockSolver;
typedef g2o::LinearSolverCSparse< BlockSolver::PoseMatrixType > LinearSolver;

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

struct g2osolver_t{
    LinearSolver* lSolver;
    BlockSolver* bSolver;
    g2o::OptimizationAlgorithmLevenberg* solver;
    g2o::SparseOptimizer globalOptimizer;

};

void match(g2osolver_t* g2oSolver, state_t* state, config_t* config, camera_param_t *camera){
    int curridx = config->read_count;
    int lastidx = curridx;
    while(1){
        curridx = config->read_count;
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
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(config->read_count);
        v->setEstimate(Eigen::Isometry3d::Identity());
        g2oSolver->globalOptimizer.addVertex(v);
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices()[0] = g2oSolver->globalOptimizer.vertex(lastidx);
        edge->vertices()[1] = g2oSolver->globalOptimizer.vertex(curridx);
        Eigen::Matrix<double, 6, 6> infoMatrix = Eigen::Matrix<double, 6, 6>::Identity();
        infoMatrix(0,0) = infoMatrix(1,1) = infoMatrix(2,2) = 100;
        infoMatrix(3,3) = infoMatrix(4,4) = infoMatrix(6,6) = 100;
        edge->setInformation(infoMatrix);
        edge->setMeasurement(trans);
        g2oSolver->globalOptimizer.addEdge(edge);
        lastidx = curridx;
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
    cout<<"optimizing pose graph, vertices: "<<g2oSolver->globalOptimizer.vertices().size()<<endl;
    g2oSolver->globalOptimizer.save("./data/result_before.g2o");
    g2oSolver->globalOptimizer.initializeOptimization();
    g2oSolver->globalOptimizer.optimize( 100 ); //可以指定优化步数
    g2oSolver->globalOptimizer.save( "./data/result_after.g2o" );
    cout<<"Optimization done."<<endl;
    g2oSolver->globalOptimizer.clear();
}


int main(int argc, char *argv[])
{
    cv::initModule_nonfree();
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

    //initialize G2O
    g2osolver_t g2oSolver;

    g2oSolver.lSolver = new LinearSolver();
    g2oSolver.lSolver->setBlockOrdering(false);
    g2oSolver.bSolver = new BlockSolver(g2oSolver.lSolver);
    g2oSolver.solver = new g2o::OptimizationAlgorithmLevenberg(g2oSolver.bSolver);
    g2oSolver.globalOptimizer.setAlgorithm(g2oSolver.solver);
    g2oSolver.globalOptimizer.setVerbose(false);

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
    state.fm->matchFrame(f1, NULL, &camera, true);
    state.cloud.addFrame(f1, NULL, &camera, true);
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(config.read_count);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed(true);
    g2oSolver.globalOptimizer.addVertex(v);
    int lastIndex = config.read_count;
    config.read_count++;


    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud(state.cloud.getCloud(), "rgbd");

    boost::thread matchThread(match, &g2oSolver, &state, &config, &camera);
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
