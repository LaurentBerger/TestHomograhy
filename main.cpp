#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

double SquareError(Mat h, vector<Point2d> src,vector<Point2d> dst)
{
    double e=0;
    for (int i = 0; i < src.size(); i++)
    {
        double x = src[i].x*h.at<double>(0,0)+src[i].y*h.at<double>(0,1)+h.at<double>(0,2);
        double y = src[i].x*h.at<double>(1,0)+src[i].y*h.at<double>(1,1)+h.at<double>(1,2);
        e += (x-dst[i].x)*(x-dst[i].x)+(y-dst[i].y)*(y-dst[i].y);
    }

return e;
}

vector<double> AjusteHomography(vector<Point2d> x,vector <Point2d> y,vector<double> paramIni);



vector<KeyPoint> kpts1,kpts2;
vector<vector<DMatch> > couple;
Mat m1;
float maxDist ;
float minDist;
const char* window_name = "Original";
int noiseLevel=0;
/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void NoisAmpl(int, void*)
{
    vector<Point2d> src,dst;
    for (int i = 1; i < couple.size(); i++)
        if (couple[i][0].distance < (maxDist + minDist) / 10)
        {
            Point2f p(rand()/double(RAND_MAX)-0.5,rand()/double(RAND_MAX)-0.5);
            src.push_back(kpts1[couple[i][0].queryIdx].pt+noiseLevel/2.0*p);
            dst.push_back(kpts2[couple[i][0].trainIdx].pt);
        }
    cout << "# matches " << src.size() << "\n";
    Mat h=findHomography(src,dst,CV_RANSAC);

    cout<<"RANSAC ="<<h<<endl;
    cout << "Error RANSAC" << SquareError(h,src,dst)<<endl;
    h=findHomography(src,dst,CV_LMEDS);
    cout<<"LMEDS="<<h<<endl;
    cout << "Error LMEDS" << SquareError(h,src,dst)<<endl;
    h=findHomography(src,dst,16);
    cout<<"SIXTEEN="<<h<<endl;
    cout << "Error SIXTEEN" << SquareError(h,src,dst)<<endl;
    vector<double> paramIni = {0,0,0,0,0,0};
    vector<double> hh=AjusteHomography(src,dst,paramIni);
    for (int i = 0; i<hh.size();i++)
        cout << hh[i] << "\t";
    cout<<endl;
    Mat hhlvb= (Mat_<double>(3,3) << hh[0],hh[1],hh[2],hh[3],hh[4],hh[5],0,0,1);
    cout << "Error NRC" << SquareError(hhlvb,src,dst)<<endl;
    h.at<double>(0,2) *=2;
    h.at<double>(1,2) *=2;
    hhlvb.at<double>(0,2) *=2;
    hhlvb.at<double>(1,2) *=2;
    cout<<"H "<<h<<"\n";

    Mat m3;
    warpPerspective(m1, m3, h, Size(m1.cols,m1.rows));
    imshow("m3",m3);
    warpPerspective(m1, m3, hhlvb, Size(m1.cols,m1.rows));
    imshow("m3NRC",m3);

//    imshow( window_name, dst );
}


int main (int argc,char **argv)
{
    m1 = imread("f:/lib/opencv/samples/data/lena.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    namedWindow( window_name, WINDOW_AUTOSIZE );
    Mat rotation = getRotationMatrix2D(Point(m1.cols/2,m1.rows/2), 90, 1);

    Mat m2;
    warpAffine(m1, m2, rotation, Size(1.5*m1.cols,1.5*m1.rows));
    imshow(window_name,m1);
    imshow("Original rotated 90°",m2);
    cout<<rotation<<endl;
    Mat m1r,m2r;
    resize(m1, m1r, Size(),0.5,0.5);
    resize(m2, m2r, Size(),0.5,0.5);

    Mat descMat1,descMat2;
    Ptr<Feature2D> orb;
    orb = ORB::create();

    orb->detectAndCompute(m1r,Mat(), kpts1, descMat1); //OK
    orb->detectAndCompute(m2r,Mat(), kpts2, descMat2); //OK
    Ptr<DescriptorMatcher> matcher;
    matcher = DescriptorMatcher::create("BruteForce-Hamming");

    matcher->knnMatch(descMat1,descMat2,couple,1);

    maxDist = couple[0][0].distance;
    minDist = couple[0][0].distance;

    for (int i = 1; i < couple.size(); i++)
    {
        if (couple[i][0].distance>=maxDist)
            maxDist=couple[i][0].distance;
        if (couple[i][0].distance<=minDist)
            minDist=couple[i][0].distance;
    }

    createTrackbar( "Noise (X2) :",window_name, &noiseLevel, 100, NoisAmpl );

    waitKey();
return 0;

}