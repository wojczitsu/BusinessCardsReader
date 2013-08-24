#ifndef ImageProcessing
#define ImageProcessing

#include "stdafx.h"

class comparator{
public:
        bool operator()(vector<Point> c1,vector<Point>c2){
               
                return (boundingRect( Mat(c1)).x<boundingRect( Mat(c2)).x);
 
        }
 
};
 
 
 
void extractContours(Mat& image,vector< vector<Point> > contours_poly);
 
void getContours(const char* filename);

#endif