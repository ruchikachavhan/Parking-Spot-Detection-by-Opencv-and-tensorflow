#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <cmath>
using namespace cv;
using namespace std;

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
	//cout<<atan(slope)*180/3.14<<" "<<intercept<<endl;
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
int main()
{
	Mat image= imread("parking_ipm4.png");
	Mat gray, canny, thresh;
	cvtColor(image, gray, CV_BGR2GRAY);
	inRange(gray, 100, 200, thresh);
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
	/*for (size_t j=0;j<mid_lines_final[3].size();++j)
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
		}*/
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
			}
		}
	}
	imshow("image", image);
	imshow("gray", gray);
	imshow("thresh", thresh);
	imshow("canny", canny);
	waitKey(0);
}
