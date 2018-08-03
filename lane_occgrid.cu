#include "image_properties.h"

//ros libraries
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Int8MultiArray.h"
#include <std_msgs/Float64.h>

//C++ libraries
#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <algorithm>

//GRANSAC header files
#include "ransac/GRANSAC_CUDA.h"
#include "ransac/AbstractModel.hpp"
#include "ransac/QuadraticModel.hpp"

//Kalman filter libraries
#include "kalman/laneTracker.hpp"
#include "kalman/laneTrackerQuadratic.cpp"

//message files containing the structure of published message of lane coefficients
#include "lane_detection/test.h"
#include "lane_detection/msg2.h"
#include "lane_detection/msg3.h"

//OpenCV libraries
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <utility>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cuda.h>
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;


int tau=7, thresh=10;
int lane_dist=2,iterations=500,min_score=200;

string WINDOW = "Occupancy-Gird";

ros::Publisher pub_Lanedata;
ros::Publisher path_pub;
ros::Publisher pub_coeff;

int stop_threshold=10;
int stop_left=0, stop_right=0;
int stop_far_left=0, stop_far_right=0;

int class_left=0, class_right=0;
int class_far_left=0, class_far_right=0;

LaneTracker lTracker, rTracker;
LaneTracker flTracker, frTracker;

Mat mask_img,kalman_img;

int area = 40;
std::vector<std::vector<Point> > extreme;
//coefficients is a 4X5 vector
//4 corresponds to 4 lanes
//Each lane has (found status, class, coefficient of y^2, coefficient of y, coefficient of 1)
//found=1 if found, else 0
//class = 0(dashed lane), 1(single solid lane), 2(double solid lane)
//std::vector<std::vector<double>> coefficients;
std::vector<std::vector<Point> > gaps_left ;
std::vector<std::vector<Point> > gaps_right;
std::vector<std::vector<Point> > gaps_far_left;
std::vector<std::vector<Point> > gaps_far_right;

std::vector<Vec4i> hierarchy;
::lane_detection::msg3 coefficients;

//logic used for sorting points based on Y-coordinate
struct myclass {
    bool operator() (vector<cv::Point> pt1, vector<cv::Point> pt2) { return (pt1[0].y < pt2[0].y);}
} myobject;

//logic for sorting
bool less_by_y(const cv::Point& lhs, const cv::Point& rhs){
  return lhs.y < rhs.y;
}

//To determine the bottom most point of a lane
double x_base(std::vector<GRANSAC::VPFloat> Parameters){
    return Parameters[0]*400*400 + Parameters[1]*400 + Parameters[2];
}

//Input: Grayscale of IPM image
//Output: Binary image containing lane candidate points
//Applies step-edge kernel with paramters "tau" (width of lane ion pixels) and "thresh" (thresholding for the step-edge-kernel score of every pixel) 
Mat lane_filter_image(Mat src)
{
    Mat Score_Image_row=Mat::zeros(src.rows, src.cols, CV_8U);
    // Mat binary_inliers_image=Mat::zeros(src.rows, src.cols, CV_8U);
    Mat binary_inliers_image_row=Mat::zeros(src.rows, src.cols, CV_8U);
    namedWindow(WINDOW,CV_WINDOW_AUTOSIZE);
    createTrackbar("tau",WINDOW, &tau, 100);
    tau = getTrackbarPos("tau",WINDOW);
    createTrackbar("thresh",WINDOW, &thresh, 255);
    thresh = getTrackbarPos("thresh",WINDOW);

    for(int j=0; j<src.rows; j++)
    {
        // Mat.ptr<float> gives pointer of the row j
        unsigned char* ptRowSrc = src.ptr<uchar>(j);
        unsigned char* ptRowSI = Score_Image_row.ptr<uchar>(j);
	
        //Step-edge kernel
        for(int i = tau; i< src.cols-tau; i++)
        {
            if(ptRowSrc[i]!=0)
            {
                int aux = 2*ptRowSrc[i];

                aux += -ptRowSrc[i-tau];
                aux += -ptRowSrc[i+tau];
                aux += -2*abs((int)(ptRowSrc[i-tau]-ptRowSrc[i+tau]));
                aux = (aux<0)?(0):(aux);
                aux = (aux>255)?(255):(aux);
               
                ptRowSI[i] = (unsigned char)aux;
            }
        }
    }
    Mat Score_Image_col=Mat::zeros(src.rows, src.cols, CV_8U);
    Mat binary_inliers_image_col=Mat::zeros(src.rows, src.cols, CV_8U);
    for(int j=0; j<src.cols; j++)
    {
        for(int i = tau; i< src.rows-tau; i++)
        {
            if(src.at<Vec3b>(i,j)[0]!=0)
            {
                int aux = 2*src.at<Vec3b>(i,j)[0];
                aux += -src.at<Vec3b>(i-tau,j)[0];
                aux += -src.at<Vec3b>(i+tau,j)[0];
                aux += -2*abs(src.at<Vec3b>(i-tau,j)[0]-src.at<Vec3b>(i+tau,j)[0]);
                aux = (aux<0)?(0):(aux);
                aux = (aux>255)?(255):(aux);
                Score_Image_col.at<Vec3b>(i,j)[0] = aux;
            }
        }
    }

    //Thresholding to form binary image. White points are lane candidate points
    binary_inliers_image_row=Score_Image_row>thresh;
    binary_inliers_image_col=Score_Image_col>thresh;
    Mat binary_inliers_image;
    bitwise_or(binary_inliers_image_row,binary_inliers_image_col,binary_inliers_image);
    return binary_inliers_image;
}


//Applies Kalman filter on quadratic lane coefficients
//Maintains two state matrices: one for left and one for right
void lane_tracking(std::vector<GRANSAC::VPFloat> Parameters_fl, std::vector<GRANSAC::VPFloat> Parameters_1,std::vector<GRANSAC::VPFloat> Parameters_2, std::vector<GRANSAC::VPFloat> Parameters_fr,int found_fl,int found_1,int found_2,int found_fr)
{
    mask_img=Mat::zeros(Size(200,400), CV_8UC1);

    std::vector<GRANSAC::VPFloat> Parameters_Left, Parameters_Right;
    std::vector<GRANSAC::VPFloat> Parameters_far_Left, Parameters_far_Right;

    lTracker.predictKalman();
    for(int k=0;k<3;k++){
      Parameters_Left.push_back(lTracker.getPredicted().at<float>(k));
    }

    rTracker.predictKalman();
    for(int k=0;k<3;k++){
      Parameters_Right.push_back(rTracker.getPredicted().at<float>(k));
    }
	
    flTracker.predictKalman();
    for(int k=0;k<3;k++){
      Parameters_far_Left.push_back(flTracker.getPredicted().at<float>(k));
    }

    frTracker.predictKalman();
    for(int k=0;k<3;k++){
      Parameters_far_Right.push_back(frTracker.getPredicted().at<float>(k));
    }

    ::lane_detection::msg2 lane_coefficients;

    if(stop_far_left < stop_threshold){
	
	::lane_detection::test par;
        par.coeff=class_far_left*1.0;
        lane_coefficients.lane_coefficients.push_back(par);

        for(int p=0;p<3;p++)
        {
          ::lane_detection::test par;
          par.coeff=Parameters_far_Left[p];
          lane_coefficients.lane_coefficients.push_back(par);
        }
        /// CLASS FAR LEFT
        if(class_far_left!=0)
        {
            for (int j = 0; j < gaps_far_left.size(); ++j)
            {
                for(int i=gaps_far_left[j][0].y;i<gaps_far_left[j][1].y;i++)
                {
                    double x=Parameters_far_Left[0]*i*i + Parameters_far_Left[1]*i + Parameters_far_Left[2];
                    if(x>0 && x<kalman_img.cols)
                    {
                        cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
            for (int j = 0; j < gaps_far_left.size(); ++j)
            {
                for(int i=gaps_far_left[j][0].x;i<gaps_far_left[j][1].x;i++)
                {
                    double y=Parameters_far_Left[0]*i*i + Parameters_far_Left[1]*i + Parameters_far_Left[2];
                    if(y>0 && y<kalman_img.cols)
                    {
                        cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
        }
        //ELSE IF FOR CLASS FAR LEFT
        else if(gaps_far_left.size()>0)
        {
            for(int i=gaps_far_left[0][0].y;i<gaps_far_left[gaps_far_left.size()-1][1].y;i++)
            {
                double x=Parameters_far_Left[0]*i*i + Parameters_far_Left[1]*i + Parameters_far_Left[2];
                if(x>0 && x<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                }
            }
            for(int i=gaps_far_left[0][0].x;i<gaps_far_left[gaps_far_left.size()-1][1].x;i++)
            {
                double y=Parameters_far_Left[0]*i*i + Parameters_far_Left[1]*i + Parameters_far_Left[2];
                if(y>0 && y<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                }
            }
        }

    if(stop_left < stop_threshold){

	::lane_detection::test par;
        par.coeff=class_left*1.0;
        lane_coefficients.lane_coefficients.push_back(par);
	
        for(int p=0;p<3;p++)
        {
          ::lane_detection::test par;
          par.coeff=Parameters_Left[p];
          lane_coefficients.lane_coefficients.push_back(par);
        }
        // CLASS LEFT
        if(class_left!=0)
        {
            for (int j = 0; j < gaps_left.size(); ++j)
            {
                for(int i=gaps_left[j][0].y;i<gaps_left[j][1].y;i++)
                {
                    double x=Parameters_Left[0]*i*i + Parameters_Left[1]*i + Parameters_Left[2];
                    if(x>0 && x<kalman_img.cols){
                        cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
            for (int j = 0; j < gaps_left.size(); ++j)
            {
                for(int i=gaps_left[j][0].x;i<gaps_left[j][1].x;i++)
                {
                    double y=Parameters_Left[0]*i*i + Parameters_Left[1]*i + Parameters_Left[2];
                    if(y>0 && y<kalman_img.cols){
                        cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
        }
        //ELSE IF FOR CLASS LEFT
        else if(gaps_left.size()>0)
        {
            for(int i=gaps_left[0][0].y;i<gaps_left[gaps_left.size()-1][1].y;i++)
            {
                double x=Parameters_Left[0]*i*i + Parameters_Left[1]*i + Parameters_Left[2];
                if(x>0 && x<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                }
            }
        for(int i=gaps_left[0][0].x;i<gaps_left[gaps_left.size()-1][1].x;i++)
            {
                double y=Parameters_Left[0]*i*i + Parameters_Left[1]*i + Parameters_Left[2];
                if(y>0 && y<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                }
            }
        }

    if(stop_right < stop_threshold){

	::lane_detection::test par;
        par.coeff=class_right*1.0;
        lane_coefficients.lane_coefficients.push_back(par);
        for(int p=0;p<3;p++)
        {
          ::lane_detection::test par;
          par.coeff=Parameters_Right[p];
          lane_coefficients.lane_coefficients.push_back(par);
        }
        //CLASS RIGHT
        if(class_right!=0)
        {
            for (int j = 0; j < gaps_right.size(); ++j)
            {
                for(int i=gaps_right[j][0].y;i<gaps_right[j][1].y;i++)
                {
                    double x=Parameters_Right[0]*i*i + Parameters_Right[1]*i + Parameters_Right[2];
                    if(x>0 && x<kalman_img.cols)
                    {
                        cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
            for (int j = 0; j < gaps_right.size(); ++j)
            {
                for(int i=gaps_right[j][0].x;i<gaps_right[j][1].x;i++)
                {
                    double y=Parameters_Right[0]*i*i + Parameters_Right[1]*i + Parameters_Right[2];
                    if(y>0 && y<kalman_img.cols)
                    {
                        cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
        }
        else if(gaps_right.size()>0)
        {
            for(int i=gaps_right[0][0].y;i<gaps_right[gaps_right.size()-1][1].y;i++)
            {
                double x=Parameters_Right[0]*i*i + Parameters_Right[1]*i + Parameters_Right[2];
                if(x>0 && x<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                }
            }
        for(int i=gaps_right[0][0].x;i<gaps_right[gaps_right.size()-1][1].x;i++)
            {
                double y=Parameters_Right[0]*i*i + Parameters_Right[1]*i + Parameters_Right[2];
                if(y>0 && y<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                }
            }
        }


    if(stop_far_right < stop_threshold){
        
	::lane_detection::test par;
        par.coeff=class_far_right*1.0;
        lane_coefficients.lane_coefficients.push_back(par);

        for(int p=0;p<3;p++)
        {
          ::lane_detection::test par;
          par.coeff=Parameters_far_Right[p];
          lane_coefficients.lane_coefficients.push_back(par);
        }
        //CLASS FAR RIGHT
        if(class_far_right!=0)
        {
            for (int j = 0; j < gaps_far_right.size(); ++j)
            {
                for(int i=gaps_far_right[j][0].y;i<gaps_far_right[j][1].y;i++)
                {
                    double x=Parameters_far_Right[0]*i*i + Parameters_far_Right[1]*i + Parameters_far_Right[2];
                    if(x>0 && x<kalman_img.cols){
                        cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
            for (int j = 0; j < gaps_far_right.size(); ++j)
            {
                for(int i=gaps_far_right[j][0].x;i<gaps_far_right[j][1].x;i++)
                {
                    double y=Parameters_far_Right[0]*i*i + Parameters_far_Right[1]*i + Parameters_far_Right[2];
                    if(y>0 && y<kalman_img.cols){
                        cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                        cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    }
                }
            }
        }
        //CLASS FAR LEFT ELSE IF
        else if(gaps_far_right.size()>0)
        {
            for(int i=gaps_far_right[0][0].y;i<gaps_far_right[gaps_far_right.size()-1][1].y;i++)
            {
                double x=Parameters_far_Right[0]*i*i + Parameters_far_Right[1]*i + Parameters_far_Right[2];
                if(x>0 && x<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(x,i), 1, cv::Scalar(255,255,255), -1);
                }
            }
            for(int i=gaps_far_right[0][0].x;i<gaps_far_right[gaps_far_right.size()-1][1].x;i++)
            {
                double y=Parameters_far_Right[0]*i*i + Parameters_far_Right[1]*i + Parameters_far_Right[2];
                if(y>0 && y<kalman_img.cols)
                {
                    cv::circle(kalman_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                    cv::circle(mask_img, Point(i,y), 1, cv::Scalar(255,255,255), -1);
                }
            }
        }

    coefficients.coefficients.push_back(lane_coefficients);

    if(found_fl==1){
        flTracker.correctKalman(Parameters_fl[0],Parameters_fl[1],Parameters_fl[2]);
        stop_far_left=0;
    }
    if(found_1==1){
        lTracker.correctKalman(Parameters_1[0],Parameters_1[1],Parameters_1[2]);
        stop_left=0;
    }
    if(found_2==1){
        rTracker.correctKalman(Parameters_2[0],Parameters_2[1],Parameters_2[2]);
        stop_right=0;
    }
    if(found_fr==1){
        frTracker.correctKalman(Parameters_fr[0],Parameters_fr[1],Parameters_fr[2]);
        stop_far_right=0;
    }

    stop_far_left+=1-found_fl;
    stop_left+=1-found_1;
    stop_right+=1-found_2;
    stop_far_right+=1-found_fr;

    pub_coeff.publish(coefficients);
}
}
}
}
}



void ransac_fit(const sensor_msgs::ImageConstPtr& msg){
    //Extract image from message
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        if (enc::isColor(msg->encoding))
            cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
        else
            cv_ptr = cv_bridge::toCvShare(msg, enc::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    //src is 400X480. Since lane_filter works better if the image is larger, we convert it to 200X240 only after passing it through lane_filter
    Mat gray,src = cv_ptr->image;
    cvtColor(src, gray, CV_BGR2GRAY);
    cv::imshow("Original",src);
    
  // rectangle( gray,       //If front cameras are used
  //          Point( 0, 0 ),
  //          Point( 400, 150),
  //          Scalar( 0,0,0 ),
  //          -1,
  //          8 );
    cv::imshow("Gray",gray);
    waitKey(1);

    //To check if we get an image
    if(gray.rows>0){
        //lane_filter takes grayscale image as input
        Mat src2=lane_filter_image(gray);
	Mat Final_Lane_Filter = Mat::zeros(src2.rows, src2.cols, CV_8UC1);
        Mat result= Mat::zeros(Size(200,400), CV_8UC1);

        int dilation_size=1;
        Mat element = getStructuringElement( MORPH_RECT,
                                           Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           Point( dilation_size, dilation_size ) );
        
        imshow("Lane filter",src2);
        //dilating to merge nearby lane blobs and prevent small lane blobs from disappearing when the image is down-sized
        dilate( src2,src2, element );

        //Area thresholding
	std::vector<std::vector<Point> > contour_area_threshold;
        findContours(src2, contour_area_threshold, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        createTrackbar("area",WINDOW, &area, 200);
        area = getTrackbarPos("area",WINDOW);
        for (int i = 0; i < contour_area_threshold.size(); ++i)
        {
            if (contourArea(contour_area_threshold[i]) >= area)
            {
                drawContours(Final_Lane_Filter, contour_area_threshold, i, Scalar(255),-1);
            }
        }
        imshow("Final_right",Final_Lane_Filter);
	
        // resize(blank,blank,Size(200,240),CV_INTER_LINEAR);  //If front cameras are used
        // copyMakeBorder(blank,blank,160,0,0,0,BORDER_CONSTANT,Scalar(0));    

        //Occupancy grid is of size 200X400 where 20 pixels corresponds to 1metre
        //So we resize the image by half and add 160 pixels padding to the top
        resize(Final_Lane_Filter,Final_Lane_Filter,Size(200,240),CV_INTER_LINEAR);
        copyMakeBorder(Final_Lane_Filter,Final_Lane_Filter,160,0,0,0,BORDER_CONSTANT,Scalar(0));

        Final_Lane_Filter.copyTo(kalman_img);
        //To store if the corresponding lanes are detected
        int found_fl=0,found_fr=0,found_1=0,found_2=0;
        //To store coefficients of quadratic equation of lane: x = ay^2 + by + c
        std::vector<GRANSAC::VPFloat> Parameters_fl, Parameters_fr, Parameters_1, Parameters_2;

	gaps_far_left.clear();
	gaps_left.clear();
	gaps_right.clear();
	gaps_far_right.clear();
        for(int n_lanes=0;n_lanes<4;n_lanes++)
        {
            std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
            // std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints_Y;

            vector<Point> locations;   // output, locations of non-zero pixels
            //Locations is the vector of all lane candidate points
            findNonZero(Final_Lane_Filter, locations);
            //Stores top-most and bottom-most candidate points for drawing lanes in the end

            //Converting locations to a data-type compatible with GRANSAC package
            for (int i = 0; i < locations.size(); ++i)
            {
                cv::Point Pt=locations[i];
                std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(Pt.x, Pt.y);
                CandPoints.push_back(CandPt);
            }
            // for (int i = 0; i < locations.size(); ++i)
            // {
            //     cv::Point Pt=locations[i];
            //     std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(Pt.y, Pt.x);
            //     CandPoints_Y.push_back(CandPt);
            // }
            
            //GRANSAC parameters
            //A candidate lane point is considered an inlier if it is within lane_dist pixels away from the quadratic line
            createTrackbar("lane_dist",WINDOW, &lane_dist, 100);
            lane_dist = getTrackbarPos("lane_dist",WINDOW);
            //Number of iterations to run RANSAC
            createTrackbar("iterations",WINDOW, &iterations, 1000);
            iterations = getTrackbarPos("iterations",WINDOW);
            //Best-fit line is considered a lane if its inliers are more than min_score
            createTrackbar("min_score",WINDOW, &min_score, 1000);
            min_score = getTrackbarPos("min_score",WINDOW);

            GRANSAC::RANSAC<Quadratic2DModel, 3> Estimator;
            //GRANSAC::RANSAC<Cubic2DModel, 4> Estimator;
            Estimator.Initialize(lane_dist,iterations); // Threshold, iterations
            Estimator.Estimate(CandPoints);
            auto BestInliers = Estimator.GetBestInliers();
            auto points_found = Estimator.get_points_found();
            GRANSAC::VPFloat BestScore = Estimator.GetBestScore();
            
            vector<cv::Point> inliers;
            vector< vector<cv::Point> > int_extreme;
            Mat intermediate = Mat::zeros(400, 200, CV_8UC1);

            //Lane classes
            int dashed=0;
            int not_lane=0;
              
            if (points_found && BestScore>min_score){
                for (auto& Inlier : BestInliers){
                    auto RPt = std::dynamic_pointer_cast<Point2D>(Inlier);
                    cv::Point Pt(floor(RPt->m_Point2D[0]), floor(RPt->m_Point2D[1]));
                    cv::circle(intermediate, Pt, 4, cv::Scalar(255,255,255), -1);
                    inliers.push_back(Pt);
                }

		std::vector<std::vector<Point> > contour_lane;
                findContours(intermediate, contour_lane, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
                for (int i = 0; i < contour_lane.size(); i++){
                    Point extTop  = *min_element(contour_lane[i].begin(), contour_lane[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
                    Point extBot  = *max_element(contour_lane[i].begin(), contour_lane[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
                    vector<cv::Point> v{extTop,extBot};
                    int_extreme.push_back(v); 
                }
                sort(int_extreme.begin(), int_extreme.end(), myobject);
                vector<int> classify;
                for(int i=1;i<int_extreme.size();i++)
                    classify.push_back(sqrt((int_extreme[i][0].y-int_extreme[i-1][1].y)*(int_extreme[i][0].y-int_extreme[i-1][1].y)+(int_extreme[i][0].x-int_extreme[i-1][1].x)*(int_extreme[i][0].x-int_extreme[i-1][1].x)));
                for(int i=0;i<classify.size();i++){
                    if(classify[i]>5 && classify[i]<15){
                        dashed++;
                    }
                    if(classify[i]>=15){
                        not_lane++;
                    }
                }
                std::vector<GRANSAC::VPFloat> Parameters =  Estimator.GetBestParameters();
                int color=255;
                if(not_lane<2){
                    if(dashed>=3){
                        color=50;
                    }
                    else{
                        bitwise_and(intermediate,Final_Lane_Filter,intermediate);
                        vector<cv::Point> v;
                        findNonZero(intermediate, v);
                        if(v.size()<160*inliers.size()/100.0){
                          color=150;
                        }
                    }

                    for(size_t i=0;i!=inliers.size();i++){
                        cv::circle(result, inliers[i], 1, cv::Scalar(color,color,color), -1);
                    }

                    if(x_base(Parameters)<50){
                        found_fl=1;
			cout<<"fl "<<n_lanes<<endl;
                        Parameters_fl=Parameters;
                        switch(color){
                            case 50: class_far_left = 0; break;
                            case 150: class_far_left = 1; break;
                            case 255: class_far_left = 2; break;
                        }
                        gaps_far_left=int_extreme;
                    }

                    else if(x_base(Parameters)<100){
                        found_1=1;
			cout<<"l "<<n_lanes<<endl;
                        Parameters_1=Parameters;
                        switch(color){
                            case 50: class_left = 0; break;
                            case 150: class_left = 1; break;
                            case 255: class_left = 2; break;
                        }
                        gaps_left=int_extreme;
                    }
                    else if(x_base(Parameters)<150){
                        found_2=1;
			cout<<"r "<<n_lanes<<endl;
                        Parameters_2=Parameters;  
                        switch(color){
                            case 50: class_right = 0; break;
                            case 150: class_right = 1; break;
                            case 255: class_right = 2; break;
                        }
                        gaps_right=int_extreme;     
                    }
                    else{
                        found_fr=1;
			cout<<"fr "<<n_lanes<<endl;
                        Parameters_fr=Parameters;  
                        switch(color){
                            case 50: class_far_right = 0; break;
                            case 150: class_far_right = 1; break;
                            case 255: class_far_right = 2; break;
                        }
                        gaps_far_right=int_extreme;     
                    }
                }
                for(int i=0;i<inliers.size();i++){
                    cv::circle(Final_Lane_Filter, inliers[i], 20, cv::Scalar(0,0,0), -1);
                }
            }
        }

        //Starting Lane Tracking
        lane_tracking(Parameters_fl,Parameters_1,Parameters_2, Parameters_fr, found_fl, found_1, found_2,found_fr );
        //Kalman prediction
        imshow("Kalman Prediction",kalman_img);
        //Mask
        imshow("Mask",mask_img);
	std::vector<std::vector<Point> > contour;
        findContours(mask_img, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        
	extreme.clear();
        //Extremes of all lane blobs to be used by Path Planning module
        for (int i = 0; i < contour.size(); i++){
            Point extTop   = *min_element(contour[i].begin(), contour[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
            Point extBot   = *max_element(contour[i].begin(), contour[i].end(),[](const Point& lhs, const Point& rhs) {return lhs.y < rhs.y;});
            vector<cv::Point> v{extTop,extBot};
            extreme.push_back(v);
        }

        //cout<<"resize"<<endl;
        //resize(result,dst,Size(200,400),CV_INTER_LINEAR);
        cout<<"imshow"<<endl;
        //Final lanes!
    
        
            imshow("RANSAC",result); 
        //waitkey required for displaying images (imshow)
        waitKey(1);
    }
}

//Making lane contours information compatible with Path Planning module
//Generating Occupany Grid
void endprocessing()
{

    nav_msgs::Path gui_path;
    geometry_msgs::PoseStamped pose;
    for (int i=0; i<extreme.size(); i++){
        pose.pose.position.x = extreme[i][0].x;
        pose.pose.position.y = extreme[i][0].y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;
        pose.pose.orientation.w = 1.0;
        //plan.push_back(pose);
        gui_path.poses.push_back(pose);
    }

    for (int i=0; i<extreme.size(); i++){
          pose.pose.position.x = extreme[i][1].x;
          pose.pose.position.y = extreme[i][1].y;
          pose.pose.position.z = 0.0;
          pose.pose.orientation.x = 0.0;
          pose.pose.orientation.y = 0.0;
          pose.pose.orientation.z = 0.0;
          pose.pose.orientation.w = 1.0;
          gui_path.poses.push_back(pose);
    }
    path_pub.publish(gui_path);
    gui_path.poses.clear();

    nav_msgs::OccupancyGrid Final_Grid;

    Final_Grid.info.map_load_time = ros::Time::now();
    Final_Grid.header.frame_id = "lane";
    Final_Grid.info.resolution = (float)map_width/(100*(float)occ_grid_widthr);
    Final_Grid.info.width = 200;
    Final_Grid.info.height = 400;

    Final_Grid.info.origin.position.x = 0;
    Final_Grid.info.origin.position.y = 0;
    Final_Grid.info.origin.position.z = 0;

    Final_Grid.info.origin.orientation.x = 0;
    Final_Grid.info.origin.orientation.y = 0;
    Final_Grid.info.origin.orientation.z = 0;
    Final_Grid.info.origin.orientation.w = 1;

    for (int i = 0; i < mask_img.rows; ++i)
    {
        for (int j = 0; j < mask_img.cols; ++j)
        {
            if ( mask_img.at<uchar>(i,j) > 0)
            {
                Final_Grid.data.push_back(2);
            }
            else
                Final_Grid.data.push_back(mask_img.at<uchar>(i,j));
        }
    }
    pub_Lanedata.publish(Final_Grid);
}

int main(int argc, char **argv)
{
    
    
    ros::init(argc, argv, "Lane_Occupancy_Grid");
    ros::NodeHandle nh;
    
    image_transport::ImageTransport it(nh);

    image_transport::Subscriber sub = it.subscribe("/camera2/image_raw", 1, ransac_fit);    
    
    pub_coeff = nh.advertise<lane_detection::msg3>("/Lane_coefficients", 1);

    pub_Lanedata = nh.advertise<nav_msgs::OccupancyGrid>("/Lane_Occupancy_Grid", 1);
    ros::Rate loop_rate(10);
    ros::NodeHandle p;
    path_pub = p.advertise<nav_msgs::Path>("/lane_coord", 1);

    flTracker.initKalman(0.0,0.0,0.0);
    lTracker.initKalman(0.0, 0.0, 50.0);
    rTracker.initKalman(0.0, 0.0, 100.0);
    frTracker.initKalman(0.0,0.0,150.0);



    while(ros::ok())
    {   
        ros::spinOnce();
        endprocessing();
        loop_rate.sleep();
    }
    ROS_INFO("videofeed::occupancygrid.cpp::No error.");
}
