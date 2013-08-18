#include "stdafx.h"
#include <iostream>
 
#include <Windows.h>
 
 

#include <opencv\cv.h>
//#include "highgui.h"
#include <opencv\ml.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <opencv2/highgui/highgui.hpp>
 
using namespace std;
using namespace cv;


class comparator{
public:
        bool operator()(vector<Point> c1,vector<Point>c2){
               
                return (boundingRect( Mat(c1)).x<boundingRect( Mat(c2)).x);
 
        }
 
};
 
 
 
void extractContours(Mat& image,vector< vector<Point> > contours_poly);
 
void getContours(const char* filename);

