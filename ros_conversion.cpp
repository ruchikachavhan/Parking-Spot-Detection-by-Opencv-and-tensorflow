#include "image_properties.h"

//ros libraries
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Int8MultiArray.h"
#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>

//C++ libraries
#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <algorithm>

//OpenCV libraries
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <utility>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
//#include <cuda.h>

#include "opencv2/calib3d/calib3d.hpp"
using namespace cv;
using namespace std;

namespace enc = sensor_msgs::image_encodings;

int tau=7, thresh=40;

string WINDOW = "Occupancy-Gird";

ros::Publisher pub_Lanedata;
ros::Publisher path_pub;

Mat Final_Parking_Filter;

int area = 15;

//logic used for sorting points based on Y-coordinate
struct myclass {
    bool operator() (vector<cv::Point> pt1, vector<cv::Point> pt2) { return (pt1[0].y < pt2[0].y);}
} myobject;

//logic for sorting
bool less_by_y(const cv::Point& lhs, const cv::Point& rhs){
  return lhs.y < rhs.y;
}


//Input: Grayscale of IPM image
//Output: Binary image containing lane candidate points
//Applies step-edge kernel with paramters "tau" (width of lane ion pixels) and "thresh" (thresholding for the step-edge-kernel score of every pixel) 
Mat parking_filter_image(Mat src){
    namedWindow(WINDOW,CV_WINDOW_AUTOSIZE);
    createTrackbar("tau",WINDOW, &tau, 100);
    tau = getTrackbarPos("tau",WINDOW);
    createTrackbar("thresh",WINDOW, &thresh, 255);
    thresh = getTrackbarPos("thresh",WINDOW);

    Mat Score_Image_row=Mat::zeros(src.rows, src.cols, CV_8U);
    Mat binary_inliers_image_row=Mat::zeros(src.rows, src.cols, CV_8U);

    for(int j=0; j<src.rows; j++){
        unsigned char* ptRowSrc = src.ptr<uchar>(j);
        unsigned char* ptRowSI = Score_Image_row.ptr<uchar>(j);
	
        //Step-edge kernel
        for(int i = tau; i< src.cols-tau; i++){
            if(ptRowSrc[i]!=0){
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
    for(int j=0; j<src.cols; j++){
        for(int i = tau; i< src.rows-tau; i++){
            if(src.at<Vec3b>(i,j)[0]!=0){
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


void parking_box(const sensor_msgs::ImageConstPtr& msg){

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
    //src is 500X600. Since lane_filter works better if the image is larger, we convert it to 200X240 only after passing it through lane_filter
    Mat gray,src = cv_ptr->image;
    cvtColor(src, gray, CV_BGR2GRAY);
    cv::imshow("Original",src);

    //Remove grid from ipm
	for(int i=0;i<gray.rows;i=i+gray.rows/12){
                for(int j=0;j<gray.cols;j++){
                        gray.at<Vec3b>(i,j)[0]=0;
                        gray.at<Vec3b>(i,j)[1]=0;
                        gray.at<Vec3b>(i,j)[2]=0;
                }
        }

        for(int i=0;i<gray.cols;i=i+gray.cols/10){
                for(int j=0;j<gray.rows;j++){
                        gray.at<Vec3b>(j,i)[0]=0;
                        gray.at<Vec3b>(j,i)[1]=0;
                        gray.at<Vec3b>(j,i)[2]=0;
                }
        }

    cv::imshow("Gray",gray);
    waitKey(1);
    //To check if we get an image
    if(gray.rows>0){
        //lane_filter takes grayscale image as input
        Mat src2=parking_filter_image(gray);
	Final_Parking_Filter = Mat::zeros(src2.rows, src2.cols, CV_8UC1);
        
        int dilation_size=1;
        Mat element = getStructuringElement( MORPH_RECT,
                                           Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           Point( dilation_size, dilation_size ) );
        
        imshow("Parking filter",src2);
        //dilating to merge nearby lane blobs and prevent small lane blobs from disappearing when the image is down-sized
        dilate( src2,src2, element );

        //Area thresholding
	std::vector<std::vector<Point> > contour_area_threshold;
	std::vector<Vec4i> hierarchy;
cout<<"o"<<" "<<endl;
        findContours(src2, contour_area_threshold, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
cout<<"p"<<endl;
        createTrackbar("area",WINDOW, &area, 100);
        area = getTrackbarPos("area",WINDOW);

        for (int i = 0; i < contour_area_threshold.size(); ++i)
        {
            if (contourArea(contour_area_threshold[i]) >= area)
            {
                drawContours(Final_Parking_Filter, contour_area_threshold, i, Scalar(255),-1);
            }
        }

        imshow("Final_right",Final_Parking_Filter);
	
        //Occupancy grid is of size 200X400 where 20 pixels corresponds to 1metre
        //So we resize the image by half and add 160 pixels padding to the top
        //resize(Final_Parking_Filter,Final_Parking_Filter,Size(200,240),CV_INTER_LINEAR);
        //copyMakeBorder(Final_Parking_Filter,Final_Parking_Filter,160,0,0,0,BORDER_CONSTANT,Scalar(0));
        //waitkey required for displaying images (imshow)
        waitKey(1);
    }
}

float find_slope(Vec4i lines)
{
	if (lines[2]-lines[0]!=0)
	{
		return atan((lines[3]-lines[1])/(lines[2]-lines[0]));
	}
	else
	{
		return 1.57;
	}
}
float max(float x, float y)
{
	if(x>=y)
	{
		return x;
	}
	else
	{
		return y;
	}
}
float  make_line(vector<float> x, vector<float> y)
{
	float x_mean=0,y_mean=0, parameters[2];
	for(size_t i=0;i<x.size();++i)
	{
		x_mean=x_mean+x[i];
	}
	x_mean=x_mean/x.size();
	for(size_t i=0;i<y.size();++i)
	{
		y_mean=y_mean+y[i];
	}
	y_mean=y_mean/y.size();
	float slope;
	float diff=0, diff_x_sq;
	for (size_t j=0;j<x.size();++j)
	{
		diff= diff + (x[j]-x_mean)*(y[j]-y_mean);
		diff_x_sq= diff_x_sq+ (x[j]-x_mean)*(x[j]-x_mean);
	}
	slope= diff/diff_x_sq;
	float intercept;
	intercept= y_mean-slope*x_mean;
	return slope;
}
float check_for_error(float distance, float min_y, float max_y)
{
	if (abs(distance-min_y)>=5)
	{
		cout<<"yes"<<endl;
		distance= abs(max_y-distance);
	}
	cout<<"d"<<distance<<endl;
	return distance;
}
Mat detect_ParkingSLots(Mat image)
{
	Mat canny, thresh;
	// cvtColor(image, gray, CV_BGR2GRAY);
	// inRange(gray, 100, 200, thresh);
	thresh = image;
	Canny(thresh, canny,100, 300,3);
	vector<Vec4i> lines, temp_lines;
	vector<vector<Vec4i> > final_lines;
	float slope, slope_t;
	int start;
	HoughLinesP(canny,  lines, 1, CV_PI / 180, 7, 10, 10);
	for (size_t i=0;i< lines.size();i++)
	{
		Vec4i l=lines[i];
		line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 3);
	}
	while(lines.size()>0)
	{
		start=0;
		Vec4i l= lines[start];
		if (l[2]-l[0]!=0)
		{
			slope=atan((l[3]-l[1])/(l[2]-l[0]));
		}
		else
		{
			slope= 1.57;
		}
		for (size_t j=0;j<lines.size();j++)
		{
			Vec4i ll=lines[j];
			if (ll[2]-ll[0]!=0)
			{
				slope_t=atan((ll[3]-ll[1])/(ll[2]-ll[0]));
			}
			else
			{
				slope_t= 1.57;
			}
			if (abs(slope-slope_t)<=0.087)
			{
				temp_lines.push_back(lines[j]);
				lines.erase(lines.begin()+j);
				j=-1;
			}
		}
		final_lines.push_back(temp_lines);
		temp_lines.clear();
	}
		vector< Vec4i> mid_lines;
		vector<vector<Vec4i> > mid_lines_final;

for (int i=0;i<final_lines.size();++i)
{
	if (final_lines[i].size()<=5)
		{
			final_lines.erase(final_lines.begin()+i);
		}
}
cout<<"final_lines"<<final_lines.size()<<endl;
int it=0,jt;
while(final_lines.size()>0)
{
		jt=0;
		while(final_lines[it].size()>0)
		{
			Vec4i l= final_lines[it][jt];
			slope= find_slope(l);
			final_lines[it].erase(final_lines[it].begin()+jt);
			for (size_t j=0;j<final_lines[it].size();j++)
			{
				Vec4i ll= final_lines[it][j];
				Vec4i temp;
				temp[0]=(l[0]+l[2])/2;
				temp[1]=(l[1]+l[3])/2;
				temp[2]=(ll[0]+ll[2])/2;
				temp[3]=(ll[1]+ll[3])/2;
				slope_t= find_slope(temp);
				float c1= l[3]-slope*l[2];
				float c2= ll[3]-slope_t*ll[2];
				if ((abs(temp[3]-temp[1])<=10)|| abs(temp[2]-temp[0])<=10 || abs(c1-c2)/sqrt(1+slope*slope)<=10)
				{
					mid_lines.push_back(ll);
					final_lines[it].erase(final_lines[it].begin()+j);
					j=-1;
				}
			}
			if(mid_lines.size()>0)
			{
				mid_lines_final.push_back(mid_lines);
				mid_lines.clear();
			}
		}
		final_lines.erase(final_lines.begin()+it);
	}
	cout<<"size"<<mid_lines_final.size()<<endl;
for (size_t i=0;i<mid_lines_final.size();++i)
{
	if(mid_lines_final[i].size()<=4)
	{
		mid_lines_final.erase(mid_lines_final.begin()+i);
		i=-1;
	}
}
cout<<"size"<<mid_lines_final.size()<<endl;
//for (size_t i=0;i<mid_lines_final.size();++i)
//{
	for (size_t j=0;j<mid_lines_final[3].size();++j)
	{
		Vec4i l=mid_lines_final[3][j];
		//cout<<find_slope(mid_lines_final[i][j])<<endl;
			line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3);
	}
	for (size_t j=0;j<mid_lines_final[4].size();++j)
	{
		Vec4i l=mid_lines_final[4][j];
		//cout<<find_slope(mid_lines_final[i][j])<<endl;
			line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,0), 3);
		}
//	}
	vector<float> x,y;
	vector<float> params, intercepts;
	float x_mean, y_mean,slopes;
	for (size_t i=0;i<mid_lines_final.size();++i)
	{
		for(size_t j=0;j<mid_lines_final[i].size();++j)
		{
			x.push_back(mid_lines_final[i][j][0]);
			x.push_back(mid_lines_final[i][j][2]);
			y.push_back(mid_lines_final[i][j][1]);
			y.push_back(mid_lines_final[i][j][3]);
		}
		x_mean=0, y_mean=0;
		for(size_t i=0;i<x.size();++i)
	{
		x_mean=x_mean+x[i];
	}
	x_mean=x_mean/x.size();
	for(size_t i=0;i<y.size();++i)
	{
		y_mean=y_mean+y[i];
	}
	y_mean=y_mean/y.size();
	slopes=make_line(x,y);
	intercepts.push_back(y_mean-x_mean*slopes);
		params.push_back(slopes);
		x.clear();
		y.clear();
	}
	float slope_1, slope_2,distance,s;
	for (size_t j=0;j<params.size();++j)
	{
		slope_1= abs(atan(params[j]));
		for(size_t k=j+1;k<params.size();++k)
		{
			slope_2=abs(atan(params[k]));
			if(abs(slope_1-slope_2)<=0.35 && slope_1*180/3.14< 80)
			{
				// for (size_t y=0;y<mid_lines_final[j].size();++y)
				// {
				// 	line(image, Point(mid_lines_final[j][y][0],mid_lines_final[j][y][1]),  Point(mid_lines_final[j][y][2],mid_lines_final[j][y][3]), Scalar(255,255,255), 3);
				// }
				cout<<mid_lines_final[j][0][0]<<" "<<slope_1*mid_lines_final[j][0][0]+intercepts[j]<<" "<<mid_lines_final[j][1][2]<<" "<<slope_1*mid_lines_final[j][1][2]+intercepts[j]<<endl;
				line(image, Point(mid_lines_final[j][0][0], slope_1*mid_lines_final[j][0][0]+intercepts[j]), Point(mid_lines_final[j][1][2], slope_1*mid_lines_final[j][1][2]+intercepts[j]), Scalar(0,0,255), 2);
				//finding minimum and maximum x coordinates of lines
				float minx=image.cols, maxx=0.0;
				for (int h=0;h<mid_lines_final[j].size();++h)
				{
						if(minx>mid_lines_final[j][h][0])
						{
							minx=mid_lines_final[j][h][0];
						}
				}
				for (int h=0;h<mid_lines_final[j].size();++h)
				{
						if(maxx<mid_lines_final[j][h][0])
						{
							maxx=mid_lines_final[j][h][0];
						}
				}
				line(image, Point(minx, slope_1*minx+intercepts[j]), Point(maxx, slope_1*maxx+intercepts[j]), Scalar(255,255,255), 3);
				// for (size_t y=0;y<mid_lines_final[k].size();++y)
				// {
				// 	line(image, Point(mid_lines_final[k][y][0],mid_lines_final[k][y][1]),  Point(mid_lines_final[k][y][2],mid_lines_final[k][y][3]), Scalar(255,255,255), 3);
				// }
				cout<<"j" <<j<<" "<<"k"<<" "<<k<<" "<<"slope"<<slope_1*180/3.14<<" "<<slope_2*180/3.14<<" "<<abs(intercepts[j]-intercepts[k])<<endl;
				s=max(params[j],params[k]);
				distance=abs(intercepts[j]-intercepts[k])/sqrt(1+s*s);
				cout<<mid_lines_final[k][0][1]-(slope_1*minx+intercepts[j]-distance)<<endl;
				//cout<<abs(mid_lines_final[k][0][1]-slope_1*minx+intercepts[j]-distance)<<endl;
				float dd= check_for_error(mid_lines_final[k][0][1],slope_1*minx+intercepts[j]-distance, slope_1*mid_lines_final[j][0][0]+intercepts[j] ) ;
				line(image, Point(minx,slope_1*minx+intercepts[j]-dd), Point(maxx,slope_1*maxx+intercepts[j]-dd), Scalar(255,255,255), 3);
				cout<<"distance"<<" "<<dd<<endl;
				float slope_perp = -1/max(slope_1, slope_2);
				line(image, Point((minx+maxx)/2, slope_perp*(minx+maxx)/2), Point((minx+maxx)/2, slope_perp*(minx+maxx)/2-distance), Scalar(255,255,255), 3);
			}
				// DRAWING PERPENDICULAR LINES
				if(abs(slope_1-slope_2)<=0.9 && slope_2*180/3.14>80)
				{
					cout<<"found"<<endl;
				// for (size_t y=0;y<mid_lines_final[j].size();++y)
				// {
				// 	line(image, Point(mid_lines_final[j][y][0],mid_lines_final[j][y][1]),  Point(mid_lines_final[j][y][2],mid_lines_final[j][y][3]), Scalar(255,255,255), 3);
				// }
				//cout<<mid_lines_final[j][0][0]<<" "<<slope_1*mid_lines_final[j][0][0]+intercepts[j]<<" "<<mid_lines_final[j][1][2]<<" "<<slope_1*mid_lines_final[j][1][2]+intercepts[j]<<endl;
				//line(image, Point(mid_lines_final[j][0][0], slope_1*mid_lines_final[j][0][0]+intercepts[j]), Point(mid_lines_final[j][1][2], slope_1*mid_lines_final[j][1][2]+intercepts[j]), Scalar(0,0,255), 2);
				//finding minimum and maximum x coordinates of lines
					float miny=image.rows, maxy=0.0;
				for (int h=0;h<mid_lines_final[k].size();++h)
				{
						if(miny>mid_lines_final[k][h][0])
						{
							miny=mid_lines_final[k][h][0];
						}
				}
				for (int h=0;h<mid_lines_final[k].size();++h)
				{
						if(maxy<mid_lines_final[k][h][0])
						{
							maxy=mid_lines_final[k][h][0];
						}
				}
				float minx1=image.cols, maxx1=0.0;
				for (int h=0;h<mid_lines_final[k].size();++h)
				{
						if(minx1>mid_lines_final[k][h][0])
						{
							minx1=mid_lines_final[k][h][0];
						}
				}
				for (int h=0;h<mid_lines_final[k].size();++h)
				{
						if(maxx1<mid_lines_final[k][h][0])
						{
							maxx1=mid_lines_final[k][h][0];
						}
				}
				cout<<max(slope_2, slope_1)*minx1+intercepts[k]<<" "<<max(slope_2, slope_1)*maxx1+intercepts[k]<<endl;
				//line(image, Point(minx1, miny), Point(minx1, maxy), Scalar(255,255,255), 3);
				Point p1,p2;
				line(image, Point(minx1, max(slope_2, slope_1)*minx1+intercepts[k]), Point(maxx1, max(slope_2, slope_1)*maxx1+intercepts[k]), Scalar(255,255,2552), 3);
				if(max(slope_2, slope_1)*minx1+intercepts[k]> image.rows)
				{
					p1.x= minx1;
					p1.y=0;
				}
				if(max(slope_2, slope_1)*maxx1+intercepts[k]>image.rows)
				{
					p2.x= maxx1;
					p2.y=image.rows;
				}
				line(image,p1, p2, Scalar(255,255,255),3);
			}
		}
	}
	//cout<<"Number of parking slot "<<" "<<number_parking_slot<<endl;
	imshow("image", image);
	// imshow("gray", gray);
	// imshow("thresh", thresh);
	imshow("canny", canny);
	//waitKey(0);
	return image;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Parking");
    ros::NodeHandle nh;
    Mat example= imread("parking_ipm4.png");
    image_transport::ImageTransport it(nh);

    image_transport::Subscriber sub_left = it.subscribe("/camera/left_ipm", 1, parking_box);    
    image_transport::Publisher parking= it.advertise("/camera/parking", 1);

    //ros::Publisher pub = nh.advertise <std_msgs::Int8>("/numbers",10);

    //std_msgs::Int8 msg;

    ros::Rate loop_rate(10);
    
    while(ros::ok())
    {   
        ros::spinOnce();
        //msg.data= detect_number_ofParkingSLots(Final_Parking_Filter);
        Mat Parking_image=detect_ParkingSLots(example) ;
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Parking_image ).toImageMsg();
        //pub.publish(msg);
        parking.publish(msg);
	loop_rate.sleep();
    }
    ROS_INFO("parking");
}
