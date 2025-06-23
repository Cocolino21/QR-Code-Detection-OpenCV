
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <queue>
#include <random>
#include "fileManager.hpp"
# define M_PI           3.14159265358979323846
std::string projectPath;

using namespace cv;
using namespace std;



void openImageTest() {
    std::string fileName = "";
    while (openFileDlg(fileName)) {
        Mat img = imread(fileName);
        imshow("Test", img);
        waitKey(0);
    }
}
void negativeImage(){
    Mat img = imread("./Images/kids.bmp", IMREAD_GRAYSCALE);
    for (int i = 0; i < img.rows ; i++)
        for (int j = 0;j<img.cols;j++) {
           img.at<uchar>(i,j) = 255 - img.at<uchar>(i,j);
        }
    imshow("negativeKids",img);
    waitKey(0);
}

void additiveFactor(){
    Mat img = imread("./Images/kids.bmp", IMREAD_GRAYSCALE);
    for (int i = 0; i< img.rows; i++)
        for (int j = 0; j<img.cols;j++){
            img.at<uchar>(i,j) = img.at<uchar>(i,j) + 50 > 255?255:img.at<uchar>(i,j)+50;
        }
    imshow("additive kids", img);
    waitKey(0);
}

void multipFactor(){
    Mat img = imread("./Images/kids.bmp", IMREAD_GRAYSCALE);
    for (int i = 0; i< img.rows; i++)
        for (int j = 0; j < img.cols; j++){
            img.at<uchar>(i,j) = img.at<uchar>(i,j)*5>255?255:img.at<uchar>(i,j)*5;
        }
    imshow("multiplicative kids", img);
    waitKey(0);
}

void colorImage() {
    Mat img = Mat(256,256,CV_8UC3);
    Vec3b white = Vec3b(255,255,255);
    Vec3b red = Vec3b(0,0,255);
    Vec3b green = Vec3b(0,255,0);
    Vec3b yellow = Vec3b(0,255,255);
    for (int i = 0; i<255; i++)
        for (int j = 0 ; j<255; j++) {
            if (i<128&&j<128)
                img.at<Vec3b>(i,j) = white;
            else if (i<128&&j>=128)
                img.at<Vec3b>(i,j) = red;
            else if (i>=128&&j<128)
                img.at<Vec3b>(i,j) = green;
            else
                img.at<Vec3b>(i,j) = yellow;
        }
    imshow("4 color image", img);
    waitKey(0);
}

void printInverseMatrix(){
    float vec[] = {1,2,-1,2,1,2,-1,2,1};
    Mat matrix = Mat(3,3,CV_32FC1,vec);
    cout<<matrix.inv()<<endl;
}


void mirrorImage(){
    Mat img = imread("./Images/kids.bmp", IMREAD_GRAYSCALE);
    Mat dst = Mat(img.rows,img.cols,CV_8UC1);
    for (int i = 0; i< img.rows; i++){
    int k = 0;
        for (int j = img.cols-1; j>=0; j--){
            dst.at<uchar>(i,k) = img.at<uchar>(i,j);
            k++;
        }
    }
    imshow(" kids", img);
    imshow("inv kids kids", dst);
    waitKey(0);
}

void mirrorImage2(){
    Mat img = imread("./Images/kids.bmp", IMREAD_GRAYSCALE);
    Mat dst = Mat(img.rows,img.cols,CV_8UC1);
    int k = 0;
    for (int i = img.rows-1; i>= 0; i--){
        for (int j = 0; j<img.cols; j++){
            dst.at<uchar>(k,j) = img.at<uchar>(i,j);

        }
        k++;
    }
    imshow(" kids", img);
    imshow("inv kids kids", dst);
    waitKey(0);
}

void mirrorImageColor(){
    Mat imgColor = imread("./Images/kids.bmp",IMREAD_COLOR);
    Mat dst = Mat(imgColor.rows,imgColor.cols,CV_8UC3);
    for (int i = 0; i< imgColor.rows; i++){
    int k = 0;
        for (int j = imgColor.cols-1; j>=0; j--){
            dst.at<Vec3b>(i,k) = imgColor.at<Vec3b>(i,j);
            k++;
        }
    }
  //  imshow(" kids", imgColor);
    imshow("inv kids kids", dst);
    waitKey(0);
}

void mirrorImageColor2(){
    Mat img = imread("./Images/kids.bmp", IMREAD_COLOR);
    Mat dst = Mat(img.rows,img.cols,CV_8UC3);
    int k = 0;
    for (int i = img.rows-1; i>= 0; i--){
        for (int j = 0; j<img.cols; j++){
            dst.at<Vec3b>(k,j) = img.at<Vec3b>(i,j);

        }
        k++;
    }
    imshow(" kids", img);
    imshow("inv kids kids", dst);
    waitKey(0);
}


void mirrorImageColorModifyChannel(){
    Mat img = imread("./Images/kids.bmp", IMREAD_COLOR);
    Mat dst = Mat(img.rows,img.cols,CV_8UC3);
    int k = 0;
    for (int i = img.rows-1; i>= 0; i--){
        for (int j = 0; j<img.cols; j++){

            dst.at<Vec3b>(k,j) = img.at<Vec3b>(i,j);
        }
        k++;
    }
    imshow(" kids", img);
    imshow("inv kids kids", dst);
    waitKey(0);
}

void testSnap() {
    chdir(projectPath.c_str());

    VideoCapture cap(0); // open the default camera (i.e. the built in web cam)
    if (!cap.isOpened()) // opening the video device failed
    {
        printf("Cannot open video capture device.\n");
        return;
    }

    Mat frame;
    char numberStr[256];
    char fileName[256];

    // video resolution
    Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
        (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    // Display window
    const char* WIN_SRC = "Src"; //window for the source frame
    namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
    moveWindow(WIN_SRC, 0, 0);

    const char* WIN_DST = "Snapped"; //window for showing the snapped frame
    namedWindow(WIN_DST, WINDOW_AUTOSIZE);
    moveWindow(WIN_DST, capS.width + 10, 0);

    char c;
    int frameNum = -1;
    int frameCount = 0;

    // Create Images directory if it doesn't exist
    struct stat st = {0};
    if (stat("Images", &st) == -1) {
        mkdir("Images", 0700);
    }

    for (;;) {
        cap >> frame; // get a new frame from camera
        if (frame.empty()) {
            printf("End of the video file\n");
            break;
        }

        ++frameNum;

        imshow(WIN_SRC, frame);
        imshow(WIN_SRC, frame);

        c = waitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
        if (c == 115){ //'s' pressed - snap the image to a file
            frameCount++;
            fileName[0] = '\0';
            sprintf(numberStr, "%d", frameCount);
            strcat(fileName, "Images/A");
            strcat(fileName, numberStr);
            strcat(fileName, ".bmp");
            bool bSuccess = imwrite(fileName, frame);
            if (!bSuccess) {
                printf("Error writing the snapped image\n");
            }
            else
                imshow(WIN_DST, frame);
        }
    }
}
void testSnap2(){
 chdir(projectPath.c_str());

    VideoCapture cap(0); // open the default camera (i.e. the built in web cam)
    if (!cap.isOpened()) // opening the video device failed
    {
        printf("Cannot open video capture device.\n");
        return;
    }

    Mat frame;
    char numberStr[256];
    char fileName[256];

    // video resolution
    Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
        (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    // Display window
    const char* WIN_SRC = "Src"; //window for the source frame
    namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
    moveWindow(WIN_SRC, 0, 0);


    char c;
    int frameNum = -1;
    int frameCount = 0;

    // Create Images directory if it doesn't exist
    struct stat st = {0};
    if (stat("Images", &st) == -1) {
        mkdir("Images", 0700);
    }

    for (;;) {
        cap >> frame; // get a new frame from camera
        if (frame.empty()) {
            printf("End of the video file\n");
            break;
        }

        ++frameNum;

        imshow(WIN_SRC, frame);
        Mat dst = Mat(frame.rows,frame.cols,CV_8UC3);
        for (int i = 0; i< frame.rows; i++){
        int k = 0;
        for (int j = frame.cols-1; j>=0; j--){
            dst.at<Vec3b>(i,k) = frame.at<Vec3b>(i,j);
            Vec3b greenTemp = dst.at<Vec3b>(i,k);
            unsigned char b = greenTemp[0];
            unsigned char g = greenTemp[1];
            unsigned char r = greenTemp[2];
            Vec3b greenTemp2 = Vec3b(greenTemp[0],greenTemp[1]+100>255?255:greenTemp[1]+100,greenTemp[2]);
            dst.at<Vec3b>(i,k) = greenTemp2;

            k++;
        }
    }
  //  imshow(" kids", imgColor);
    imshow("inv kids kids", dst);

        c = waitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }



    }
}

void convColorTo3GrayScale(){
    Mat img = imread("./Images/kids.bmp", IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    Mat bDst = Mat(rows,cols,CV_8UC1);
    Mat gDst = Mat(rows,cols,CV_8UC1);
    Mat rDst = Mat(rows,cols,CV_8UC1);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j< img.cols; j++) {
            Vec3b pixel = img.at<Vec3b>(i,j);
            bDst.at<uchar>(i,j) = pixel[0];
            gDst.at<uchar>(i,j) = pixel[1];
            rDst.at<uchar>(i,j) = pixel[2];
        }
    imshow("blue channel", bDst);
    imshow("green channel",gDst);
    imshow("red channel",rDst);
    waitKey(0);
}

void convColorToGrayScale(){
    Mat img = imread("./Images/kids.bmp",IMREAD_COLOR);
    Mat dst = Mat(img.rows,img.cols,CV_8UC1);
    for (int i=0;i<img.rows;i++)
        for (int j = 0;j<img.cols;j++) {
            uchar b = img.at<Vec3b>(i,j)[0];
            uchar g = img.at<Vec3b>(i,j)[1];
            uchar r = img.at<Vec3b>(i,j)[2];
            uchar gscale = (b+g+r)/3;
            dst.at<uchar>(i,j) = gscale;
        }
    imshow("img normal", img);
    imshow("img grayscale", dst);
    waitKey(0);
}
void convColorToGrayScale2(){
    Mat img = imread("./Images/kids.bmp",IMREAD_COLOR);
    Mat dst = Mat(img.rows,img.cols,CV_8UC1);
    for (int i=0;i<img.rows;i++)
        for (int j = 0;j<img.cols;j++) {
            uchar b = img.at<Vec3b>(i,j)[0];
            uchar g = img.at<Vec3b>(i,j)[1];
            uchar r = img.at<Vec3b>(i,j)[2];
            uchar gscale = 0.29*r + 0.58*g + 0.12*b;
            dst.at<uchar>(i,j) = gscale;
        }
    imshow("img normal", img);
    imshow("img grayscale", dst);
    waitKey(0);
}

Mat treshold(Mat src, uchar prag){
    Mat dst = Mat(src.rows,src.cols,CV_8UC1);
    for (int i=0;i<src.rows;i++)
        for (int j = 0;j<src.cols;j++) {
            dst.at<uchar>(i,j)=src.at<uchar>(i,j)<=prag?0:255;
        }
    return dst;
}

void convDstToBinary(){
    Mat img = imread("./Images/kids.bmp",IMREAD_GRAYSCALE);
    Mat dst = treshold(img,170);
    imshow("img gray", img);
    imshow("img binara", dst);
    waitKey(0);
}

float max(float a,float b,float c) {
    float max = a;
    if (b>max)
        max = b;
    if (c>max)
        max = c;
    return max;
}

float min(float a,float b, float c) {
    float min = a;
    if (b<min)
        min = b;
    if (c<min)
        min = c;
    return min;
}

void convRGBToHSV(){
      Mat img = imread("./Images/kids.bmp",IMREAD_COLOR);
        Mat dst = Mat(img.rows,img.cols,CV_8UC3);
        Mat dstH = Mat(img.rows,img.cols,CV_8UC1);
        Mat dstS = Mat(img.rows,img.cols,CV_8UC1);
        Mat dstV = Mat(img.rows,img.cols,CV_8UC1);
        for (int i=0;i<img.rows;i++)
            for (int j = 0;j<img.cols;j++) {
                uchar BMare = img.at<Vec3b>(i,j)[0];
                uchar GMare = img.at<Vec3b>(i,j)[1];
                uchar RMare = img.at<Vec3b>(i,j)[2];
                float r  = (float)RMare/255 ;
                float g  = (float)GMare/255 ;
                float b  = (float)BMare/255 ;
                float M = max(r,g,b);
                float m = min(r,g,b);
                float C = M-m;
                float V = M;
                float S=0,H=0;
                if (V!=0)
                    S=C/V;
                else
                    S=0;
                if (C!=0){
                    if (M == r) H = 60 * (g - b) / C;
                    if (M == g) H = 120 + 60 * (b - r) / C;
                    if (M == b) H = 240 + 60 * (r - g) / C;
                }
                else
                    H=0;
                if (H<0)
                    H=H+360;
                float H_norm = H*255/360;
                float S_norm = S*255;
                float V_norm= V*255;
                dst.at<Vec3b>(i,j)[0] = (uchar)H_norm;
                dstH.at<uchar>(i,j) = (uchar)H_norm;
                dst.at<Vec3b>(i,j)[1] = (uchar)S_norm;
                dstS.at<uchar>(i,j) = (uchar)S_norm;
                dst.at<Vec3b>(i,j)[2] = (uchar)V_norm;
                dstV.at<uchar>(i,j) = (uchar)H_norm;
            }
    imshow("rgb",img);
    imshow("H",dstH);
    imshow("S",dstS);
    imshow("V",dstV);
    imshow("hsv",dst);
    waitKey(0);

}

void computeHistogram(std:: string fname, int* histogram, const int len) {
    Mat src = imread(fname,IMREAD_GRAYSCALE);
    int k=0;
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++){
            uchar val = src.at<uchar>(i,j);
            if (val<len) {
                histogram[val]++;
            }
        }
}
void computeHistogram(std:: string fname, float* histogram, const int len) {
    Mat src = imread(fname,IMREAD_GRAYSCALE);
    int k=0;
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++){
            uchar val = src.at<uchar>(i,j);
            if (val<len) {
                histogram[val]++;
            }
        }
    int M = src.rows*src.cols;
    for (int i = 0; i < len; i++) {
        histogram[i] = histogram[i]/(float)M;
    }
}

void showHistogram(const std:: string& name, const int* hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height,hist_cols,CV_8UC3,CV_RGB(255,255,255));
    //computes hist max
    int max_hist = 0;
    for (int i=0;i<hist_cols;i++)
        if (hist[i]>max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height/max_hist;
    int baseline = hist_height-1;
    for (int x = 0; x< hist_cols; x++) {
        Point p1 = Point(x,baseline);
        Point p2 = Point(x,baseline-cvRound(hist[x]*scale));
        line(imgHist,p1,p2,CV_RGB(255,0,255));
    }
    imshow(name, imgHist);
    waitKey(0);
}
void showHistogram(const std:: string& name, const float* hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height,hist_cols,CV_8UC3,CV_RGB(255,255,255));
    //computes hist max
    float max_hist = 0;
    for (int i=0;i<hist_cols;i++)
        if (hist[i]>max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height/max_hist;
    float baseline = hist_height-1;
    for (int x = 0; x< hist_cols; x++) {
        Point p1 = Point(x,baseline);
        Point p2 = Point(x,baseline-cvRound(hist[x]*scale));
        line(imgHist,p1,p2,CV_RGB(255,0,255));
    }
    imshow(name, imgHist);
    waitKey(0);
}

int Ifunc(uchar px){
    return px==0?1:0;
}

bool isInside(int x, int y, Mat img) {
    if (x<img.rows&&x>0&&y<img.cols&&y>0)
        return true;
    return false;
}

void geometry(const std:: string& fname) {
    int area = 0 ;
    int ri_temp = 0, ci_temp =0;
    Mat src = imread(fname,IMREAD_GRAYSCALE);
    Mat img1 (src.rows,src.cols,CV_8UC1,Scalar(255));
    Mat img2 (src.rows,src.cols,CV_8UC1,Scalar(255));
    for (int i=0;i<src.rows;i++)
        for (int j=0;j<src.cols;j++) {
            area+=Ifunc(src.at<uchar>(i,j));
            ri_temp+= i * Ifunc(src.at<uchar>(i,j));
            ci_temp+= j * Ifunc(src.at<uchar>(i,j));
        }
    int ri = ri_temp/area;
    int ci = ci_temp/area;
    int perim = 0;
    cout<<"area is : "<<area<<endl;
    cout<<"ri = "<<ri<<" ci = "<<ci<<endl;
    int x_neighb[4] = {-1,0,1,0};
    int y_neighb[4] = {0,-1,0,1};
    for (int i=0;i<src.rows;i++)
        for (int j=0;j<src.cols;j++) {
            if (src.at<uchar>(i,j)==0){
                bool ok = false;
                for (int k = 0 ; k < 4; k++) {
                    if (isInside(i+x_neighb[k],j+y_neighb[k],src)) {
                        if (src.at<uchar>((i+x_neighb[k]),(j+y_neighb[k]))==255){
                            ok=true;
                        }
                    }
                }
                if (ok)
                    perim++;
            }
        }
    cout<<"perim is : "<<perim<<endl;
    float T = 4.0*3.14*((float)area/(float)(perim*perim));
    cout<<"T is : "<<T<<endl;
    int cmax = -1;
    int cmin = src.cols+1;
    int rmin = src.rows+1;
    int rmax = -1;


    for (int i=0;i<src.rows;i++)
        for (int j=0;j<src.cols;j++) {
            if (Ifunc(src.at<uchar>(i,j))) {
                if (j>cmax)
                    cmax = j;
                if (j<cmin)
                    cmin = j;
                if (i>rmax)
                    rmax = i;
                if (i<rmin)
                    rmin = i;
            }
        }
    int C1 = cmax - cmin + 1;
    int R1 = rmax - rmin + 1;
    float R = (float)C1/R1;
    cout<<"R = "<<R<<endl;

    int hi[src.rows];
    for (int i = 0; i < src.rows; i++) {
        hi[i] = 0;
        for (int j = 0; j < src.cols; j++) {
            hi[i]+=Ifunc(src.at<uchar>(i,j));
        }
    }
    int vi[src.cols];
    for (int j=0;j<src.cols;j++) {
        vi[j] = 0;
        for (int i=0;i<src.rows;i++) {
            vi[i]+=Ifunc(src.at<uchar>(i,j));
        }
    }

    for (int i = 0; i < src.rows; i++) {
        int cnt = hi[i];
        for (int j=0;j<src.cols;j++) {
            if (cnt!=0) {
                img1.at<uchar>(i,j) = 0;
                cnt--;
            }
        }
    }

     for (int j = 0; j < src.cols; j++) {
        int cnt = vi[j];
        for (int i=0;i<src.rows;i++) {
            if (cnt!=0) {
                img2.at<uchar>(i,j) = 0;
                cnt--;
            }
        }
    }

    imshow("horiz",img1);
    waitKey(0);
    imshow("vertical",img2);
    waitKey(0);

}

void labelMatrix(const std:: string& fname) {
    Mat src = imread(fname,IMREAD_GRAYSCALE);
    Mat labels = Mat(src.rows,src.cols,CV_32SC1);
    Mat colorMatrix = Mat(src.rows, src.cols, CV_8UC3);
    labels.setTo(0);
    int noLabel = 0;
    int di[8]  = {-1,-1,-1,0,0,1,1,1};
    int dj[8] = {-1,0,1,-1,1,-1,0,1};
    for (int i = 0 ; i < src.rows; i ++)
        for (int j = 0; j < src.cols ; j++) {
            if (src.at<uchar>(i,j)==0&&labels.at<int>(i,j)==0) {
                noLabel++;
                queue<Point2i> Q;
                labels.at<int>(i,j) = noLabel;
                Q.push({i,j});
                while (!Q.empty()) {
                    Point2i pnt = Q.front();
                    Q.pop();
                    for (int k=0;k<8;k++) {
                        int ii = pnt.x + di[k];
                        int jj = pnt.y + dj[k];
                        if (isInside(ii,jj,src)&&src.at<uchar>(ii,jj)==0&&labels.at<int>(ii,jj)==0)
                            labels.at<int>(ii,jj) = noLabel;
                            Q.push({ii,jj});
                    }
                }
            }
        }
    default_random_engine gen;
    uniform_int_distribution<int> d(0,255);
    std::vector<Vec3b> colors;
    for (int i=1;i<=noLabel;i++) {
        Vec3b color;
        color[0] = d(gen);
        color[1] = d(gen);
        color[2] = d(gen);
        colors.push_back(color);
    }

    for (int i=0;i<src.rows;i++)
        for (int j=0;j<src.cols;j++) {
            colorMatrix.at<Vec3b>(i,j) = colors[labels.at<int>(i,j)];
        }

    printf("In imagine sunt %d obiecte\n", noLabel);
    imshow("SRC", src);
    imshow("Rezultat Etichetare", colorMatrix);
    waitKey(0);
}

Mat dilatare(Mat src) {
    int di[8]  = {-1,-1,-1,0,0,1,1,1};
    int dj[8] = {-1,0,1,-1,1,-1,0,1};
    Mat dst (src.rows, src.cols, CV_8UC1, Scalar(255));

    for (int i = 0; i < src.rows; i++ )
        for (int j = 0 ; j< src.cols;j++) {
            if (src.at<uchar>(i,j) == 0) {
                for (int k = 0 ; k < 8 ; k++) {
                    int new_i = i + di[k];
                    int new_j = j + dj[k];
                    if (isInside(new_i, new_j, src)) {
                        dst.at<uchar>(new_i, new_j) = 0;
                    }
                }
            }
        }

    return dst;
}

Mat eroziune(Mat src) {
    int di[8]  = {-1,-1,-1,0,0,1,1,1};
    int dj[8] = {-1,0,1,-1,1,-1,0,1};
     Mat dst (src.rows, src.cols, CV_8UC1, Scalar(255));
    for (int i = 0 ; i < src.rows; i++)
        for (int j = 0 ; j < src.cols; j++) {
            dst.at<uchar>(i,j) = src.at<uchar>(i,j);
        }
    for (int i = 0; i < src.rows; i++ )
        for (int j = 0 ; j< src.cols;j++) {
            if (src.at<uchar>(i,j)==0) {
                for (int k = 0 ; k < 8 ; k++) {
                    int new_i = i + di[k];
                    int new_j = j + dj[k];
                    if (isInside(new_i, new_j, src)&&src.at<uchar>(new_i,new_j)==255) {
                        dst.at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }


    return dst;
}

Mat extractContour(Mat src) {
    Mat eroded = eroziune(src);
    Mat dst (src.rows, src.cols, CV_8UC1, Scalar(255));
    for (int i = 0 ; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i,j)==0 && eroded.at<uchar>(i,j)!=0) {
                dst.at<uchar>(i,j) = 0;
            }
            else if (src.at<uchar>(i,j) == 0 && eroded.at<uchar>(i,j)==0) {
                dst.at<uchar>(i,j) = 255;
            }

        }

}

bool diffMatrix(Mat a, Mat b) {
    for (int i = 0; i< a.rows; i++)
        for (int j = 0; j < a.cols;j++) {
            if (a.at<uchar>(i,j)!=b.at<uchar>(i,j)){
                return true;
            }
        }
    return false;
}

Mat inverse(Mat src) {
    Mat dst(src.rows, src.cols, CV_8UC1, Scalar(255));
    for (int i = 0 ; i < src.rows; i ++) {
        for (int j = 0 ; j < src.cols ; j++) {
            if (src.at<uchar>(i,j) == 0)
                dst.at<uchar>(i,j) = 255;
            else {
                dst.at<uchar>(i,j) = 0;
            }
        }
    }
    return dst;
}

Mat intersection(Mat a, Mat b) {
    Mat dst (a.rows, a.cols, CV_8UC1, Scalar(255));
    for (int i = 0 ; i < a.rows; i++)
        for (int j = 0 ; j < a.cols; j++) {
            if (a.at<uchar>(i,j)==0&&b.at<uchar>(i,j)==0) {
                dst.at<uchar>(i,j) = 0;
            }
            else {
                dst.at<uchar>(i,j) = a.at<uchar>(i,j);
            }
        }
    return dst;
}

Mat reunion(Mat a, Mat b) {
    Mat dst (a.rows, a.cols, CV_8UC1, Scalar(255));
    for (int i = 0 ; i < a.rows; i++)
        for (int j = 0 ; j < a.cols; j++) {
            if (a.at<uchar>(i,j) == 0 || b.at<uchar>(i,j)==0)
                dst.at<uchar>(i,j) = 0;
            else {
                dst.at<uchar>(i,j) = 255;
            }
        }
    return dst;
}

Mat fillObject(Mat src) {
    Mat contour = extractContour(src);
    Mat temp (src.rows, src.cols, CV_8UC1, Scalar(255));
    Mat newt (src.rows, src.cols, CV_8UC1, Scalar(255));
    // 51, 21
    newt.at<uchar>(51,21) = 0;
    while (diffMatrix(temp, newt)) {

      //  cout<<"ok"<<endl;
           for (int i = 0 ; i < temp.rows; i++)
            for (int j = 0 ; j < temp.cols; j++) {
                temp.at<uchar>(i,j) = newt.at<uchar>(i,j);
            }
        Mat aux (newt.rows, newt.cols, CV_8UC1);
        for (int i = 0 ; i < newt.rows; i++)
            for (int j = 0 ; j < newt.cols; j++) {
                aux.at<uchar>(i,j) = newt.at<uchar>(i,j);
            }
        newt = dilatare(temp);
        newt = intersection(newt, inverse(src));



    }

    imshow("orig", src);
    imshow("filled", temp);
    waitKey(0);

    return newt;

}

int* compute_histogram(Mat src) {
    int *hist = new int[255];
    for (int i = 0 ; i < 255; i ++) {
        hist[i] = 0;
    }
    for (int i = 0 ; i < src.rows; i++)
        for (int j = 0 ; j < src.cols; j++) {
            hist[src.at<uchar>(i,j)] += 1;
        }

    return hist;
}

float medium_intensity(Mat src) {
    int *hist = compute_histogram(src);
    int sum = 0;
    for (int i = 0 ; i < 255; i++) {
        sum+=i * hist[i];
    }
    return (float)sum/(float)(src.rows * src.cols);
}



Mat contrast(Mat src, int gOutMin, int gOutMax) {
    int gInMin = INT_MAX;
    int gInMax = INT_MIN;
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i,j)<gInMin) {
                gInMin = src.at<uchar>(i,j);
            }
            if (src.at<uchar>(i,j)>gInMax) {
                gInMax = src.at<uchar>(i,j);
            }
        }
    }

    Mat dst (src.rows, src.cols, CV_8UC1);
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i,j) = (float)gOutMin + ((float)src.at<uchar>(i,j) - (float)gInMin) * (float)(gOutMax-gOutMin)/(float)(gInMax-gInMin);
        }

    }
    imshow("orig",src);
    imshow("dst",dst);
    waitKey(0);
    return dst;
}

Mat equalizeHistogram(Mat src) {
    int hist[255] = { 0 };
    for (int i = 0 ; i < 255 ; i ++ ) {
        for (int j = 0 ; j < 255; j++) {
            hist[src.at<uchar>(i,j)] += 1;
        }
    }
    float accHist[255] = { 0 };
    for (int i = 0 ; i < 255; i++) {
        int sum = 0;
        for (int j = i; j >= 0; j--) {
            sum+=hist[j];
        }
        accHist[i] = (float)sum/255;
    }
    Mat dst (src.rows, src.cols, CV_8UC1);
    for (int i = 0 ; i < src.rows; i++)
        for (int j = 0 ; j < src.cols; j++) {
            dst.at<uchar>(i,j) = 255 * accHist[src.at<uchar>(i,j)];
        }

    imshow("orig", src);
    imshow("histss", dst);
    waitKey(0);
    return dst;
}


void convolTreceJos(Mat src, int nucleu[3][3]) {

    Mat dst (src.rows, src.cols, CV_8UC1, Scalar(0));
    int dx[9] = {-1,-1,-1,0,0,0,1,1,1};
    int dy[9] = {-1,0,1,-1,0,1,-1,0,1};
    for (int i = 1 ; i < src.rows-1; i++)
        for (int j = 1 ; j < src.cols-1; j++) {
            int sum = 0;
            int small_sum = 0;
            int x = 0;
            for (int k = 0 ; k < 3; k++) {
                for (int l = 0 ; l < 3; l++) {
                    sum+=(nucleu[k][l]*src.at<uchar>(i+dx[x],j+dy[x]));
                    x++;
                    small_sum += nucleu[k][l];
                }
            }
            dst.at<uchar>(i,j) = sum/small_sum;
        }
    imshow("orig", src);
    imshow("treceJos", dst);
    waitKey(0);

}


void convolTreceSus(Mat src, int nucleu[3][3]) {
     Mat dst_temp (src.rows, src.cols, CV_32SC1, Scalar(0));
    int dx[9] = {-1,-1,-1,0,0,0,1,1,1};
    int dy[9] = {-1,0,1,-1,0,1,-1,0,1};
    for (int i = 1 ; i < src.rows-1; i++)
        for (int j = 1 ; j < src.cols-1; j++) {
            int sum = 0;
            int x = 0;
            for (int k = 0 ; k < 3; k++) {
                for (int l = 0 ; l < 3; l++) {
                    sum+=(nucleu[k][l]*src.at<uchar>(i+dx[x],j+dy[x]));
                    x++;
                }
            }
            dst_temp.at<int>(i,j) = sum;
        }

    int gInMin = INT_MAX;
    int gInMax = INT_MIN;
    int gOutMin = 0;
    int gOutMax = 255;
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (dst_temp.at<int>(i,j)<gInMin) {
                gInMin = dst_temp.at<int>(i,j);
            }
            if (dst_temp.at<int>(i,j)>gInMax) {
                gInMax = dst_temp.at<int>(i,j);
            }
        }
    }

    Mat dst (src.rows, src.cols, CV_8UC1,Scalar(0));
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i,j) = (float)gOutMin + ((float)dst_temp.at<int>(i,j) - (float)gInMin) * (float)(gOutMax-gOutMin)/(float)(gInMax-gInMin);
        }int gInMin = INT_MAX;
    int gInMax = INT_MIN;
    int gOutMin = 0;
    int gOutMax = 255;
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (dst_temp.at<int>(i,j)<gInMin) {
                gInMin = dst_temp.at<int>(i,j);
            }
            if (dst_temp.at<int>(i,j)>gInMax) {
                gInMax = dst_temp.at<int>(i,j);
            }
        }
    }

    Mat dst (src.rows, src.cols, CV_8UC1,Scalar(0));
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i,j) = (float)gOutMin + ((float)dst_temp.at<int>(i,j) - (float)gInMin) * (float)(gOutMax-gOutMin)/(float)(gInMax-gInMin);
        }

    }

    }

    imshow("orig", src);
    imshow("trece_sus", dst);
    waitKey(0);
}


void spatialFilters(Mat src) {
    int dx[9] = {-1,-1,-1,0,0,0,1,1,1};
    int dy[9] = {-1,0,1,-1,0,1,-1,0,1};
    Mat min (src.rows, src.cols, CV_8UC1, Scalar(255));
    Mat max (src.rows, src.cols, CV_8UC1, Scalar(255));
    Mat med (src.rows, src.cols, CV_8UC1, Scalar(255));
    vector<int> vec;
    for (int i = 1 ; i < src.rows-1; i++) {
        for (int j = 1 ; j < src.cols-1; j++) {
            vec.clear();
            for (int k = 0 ; k < 9 ; k++) {
                vec.push_back(src.at<uchar>(i+dx[k],j+dy[k]));
            }
            ranges::sort(vec);
            min.at<uchar>(i,j) = vec.front();
            max.at<uchar>(i,j) = vec.back();
            med.at<uchar>(i,j) = vec[vec.size()/2];
        }
    }
    imshow("orig",src);
        imshow("min",min);
        imshow("med",med);
        imshow("max",max);
    waitKey(0);

}

void gauss(Mat src) {
    int n = 5;
    float nucleu[n][n];
    float sigma = (float)n/6;

    int dx[9] = {-1,-1,-1,0,0,0,1,1,1};
    int dy[9] = {-1,0,1,-1,0,1,-1,0,1};

    for (int i = 0 ; i < n ; i++)
        for (int j = 0 ; j < n; j++) {
            nucleu[i][j] = (1/(2*CV_PI*sigma*sigma)) * exp((-((i-(n/2))*(i-(n/2))+(j-(n/2))*(j-(n/2))))/2*sigma*sigma);
        }

    Mat dst_temp(src.rows,src.cols, CV_32FC1, Scalar(0));
    for (int i = n/2; i < src.rows-n/2; i++)
        for (int j = n/2; j < src.cols-n/2; j++) {
            int x = 0;
            float sum = 0;
            for (int k = 0 ; k < 3; k++) {
                for (int l = 0 ; l < 3; l++) {
                    sum+=(nucleu[k][l]*(float)src.at<uchar>(i+dx[x],j+dy[x]));
                    x++;
                }
            }
            dst_temp.at<float>(i,j) = sum;
        }

    float gInMin = INT_MAX;
    float gInMax = INT_MIN;
    float gOutMin = 0;
    float gOutMax = 255;
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (dst_temp.at<float>(i,j)<gInMin) {
                gInMin = dst_temp.at<float>(i,j);
            }
            if (dst_temp.at<float>(i,j)>gInMax) {
                gInMax = dst_temp.at<float>(i,j);
            }
        }
    }

    Mat dst (src.rows, src.cols, CV_8UC1,Scalar(0));
    for (int i = 0 ; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i,j) = (float)gOutMin + ((float)dst_temp.at<float>(i,j) - (float)gInMin) * (float)(gOutMax-gOutMin)/(float)(gInMax-gInMin);
        }
    }

    imshow("orig", src);
    imshow("gauss",dst);
    waitKey(0);
}



/// PROJECT CODE //////////////////////////////////////////////////////////////////////////////////////////


float tolerance = 0.4f;

Point refineCenter(const Mat& img, int x, int y, int patternSize) {
    int sum_x = 0, sum_y = 0, count = 0;
    int halfSize = patternSize / 2;

    int startX = max(0, x - halfSize);
    int endX = min(img.cols - 1, x + halfSize);
    int startY = max(0, y - halfSize);
    int endY = min(img.rows - 1, y + halfSize);

    for (int cy = startY; cy <= endY; cy++) {
        for (int cx = startX; cx <= endX; cx++) {
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx1 = cx + dx;
                    int ny1 = cy + dy;
                    int nx2 = cx - dx;
                    int ny2 = cy - dy;

                    if (nx1 >= 0 && nx1 < img.cols && ny1 >= 0 && ny1 < img.rows &&
                        nx2 >= 0 && nx2 < img.cols && ny2 >= 0 && ny2 < img.rows) {
                        uchar p1 = img.at<uchar>(ny1, nx1);
                        uchar p2 = img.at<uchar>(ny2, nx2);
                        if (p1 != p2) {
                            sum_x += cx;
                            sum_y += cy;
                            count++;
                        }
                    }
                }
            }
        }
    }

    if (count > 0) {
        return Point(sum_x / count, sum_y / count);
    }

    return Point(x, y);
}

Mat dynamicThresholdBinarization(Mat src, Mat& debugImage) {
    if (src.channels() != 1) {
        cout << "Error: Input image must be grayscale" << endl;
        return src;
    }

    cvtColor(src, debugImage, COLOR_GRAY2BGR);

    int T_i = 128;
    int T1 = 0;
    int error = 2;

    Mat dst = Mat(src.rows, src.cols, CV_8UC1);

    bool converged = false;
    int iterations = 0;
    int maxIterations = 20;

    putText(debugImage, "Dynamic Thresholding Process", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
    putText(debugImage, "Initial Threshold: " + to_string(T_i), Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    while (!converged && iterations < maxIterations) {
        long long sum_below = 0;
        int count_below = 0;
        long long sum_above = 0;
        int count_above = 0;

        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                uchar pixel = src.at<uchar>(i, j);
                if (pixel < T_i) {
                    sum_below += pixel;
                    count_below++;
                } else {
                    sum_above += pixel;
                    count_above++;
                }
            }
        }

        float V1 = (count_below > 0) ? (float)sum_below / count_below : 0;
        float V2 = (count_above > 0) ? (float)sum_above / count_above : 255;

        T1 = (int)((V1 + V2) / 2);

        string iterText = "Iter " + to_string(iterations + 1) + ": T=" + to_string(T1) +
                         " V1=" + to_string((int)V1) + " V2=" + to_string((int)V2);
        putText(debugImage, iterText, Point(10, 90 + iterations * 25),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

        if (abs(T1 - T_i) <= error) {
            converged = true;
            putText(debugImage, "CONVERGED!", Point(10, 90 + iterations * 25 + 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        }

        T_i = T1;
        iterations++;
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = (src.at<uchar>(i, j) < T_i) ? 0 : 255;
        }
    }

    putText(debugImage, "Final Threshold: " + to_string(T1), Point(10, src.rows - 60),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    putText(debugImage, "Iterations: " + to_string(iterations), Point(10, src.rows - 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

    cout << "Final threshold: " << T1 << " (iterations: " << iterations << ")" << endl;

    return dst;
}

bool checkPatternRatio(const vector<int>& runs, int startIdx, Mat& debugImage, int y, string direction) {
    if (startIdx + 4 >= runs.size()) return false;

    vector<int> pattern(runs.begin() + startIdx, runs.begin() + startIdx + 5);

    int minLength = *min_element(pattern.begin(), pattern.end());

    if (minLength < 1) return false;

    vector<float> ratios;
    for (int run : pattern) {
        ratios.push_back((float)run / minLength);
    }

    vector<float> expectedRatios = {1.0f, 1.0f, 3.0f, 1.0f, 1.0f};

    bool isValid = true;
    for (int i = 0; i < 5; i++) {
        if (abs(ratios[i] - expectedRatios[i]) > tolerance) {
            isValid = false;
            break;
        }
    }

    if (isValid && debugImage.rows > 0) {
        string ratioText = direction + " Pattern: ";
        for (int i = 0; i < 5; i++) {
            ratioText += to_string((int)(ratios[i] * 10) / 10.0f);
            if (i < 4) ratioText += ":";
        }

        int textY = min(max(y, 20), debugImage.rows - 20);
        putText(debugImage, ratioText, Point(10, textY),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
    }

    return isValid;
}

vector<int> getHorizontalRuns(const Mat& img, int y) {
    vector<int> runs;
    if (y < 0 || y >= img.rows) return runs;

    int currentRun = 1;
    uchar lastPixel = img.at<uchar>(y, 0);

    for (int x = 1; x < img.cols; x++) {
        uchar pixel = img.at<uchar>(y, x);
        if (pixel == lastPixel) {
            currentRun++;
        } else {
            runs.push_back(currentRun);
            currentRun = 1;
            lastPixel = pixel;
        }
    }

    if (currentRun > 0) {
        runs.push_back(currentRun);
    }

    return runs;
}

vector<int> getVerticalRuns(const Mat& img, int x) {
    vector<int> runs;
    if (x < 0 || x >= img.cols) return runs;

    int currentRun = 1;
    uchar lastPixel = img.at<uchar>(0, x);

    for (int y = 1; y < img.rows; y++) {
        uchar pixel = img.at<uchar>(y, x);
        if (pixel == lastPixel) {
            currentRun++;
        } else {
            runs.push_back(currentRun);
            currentRun = 1;
            lastPixel = pixel;
        }
    }

    if (currentRun > 0) {
        runs.push_back(currentRun);
    }

    return runs;
}

bool isFinderPattern(const Mat& binaryImage, int centerX, int centerY, Mat& debugImage) {
    if (centerX < 0 || centerX >= binaryImage.cols ||
        centerY < 0 || centerY >= binaryImage.rows)
        return false;

    if (binaryImage.at<uchar>(centerY, centerX) != 0) {
        return false;
    }

    circle(debugImage, Point(centerX, centerY), 3, Scalar(255, 0, 255), 2);
    putText(debugImage, "?", Point(centerX + 5, centerY - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

    vector<int> horizontalRuns = getHorizontalRuns(binaryImage, centerY);
    vector<int> verticalRuns = getVerticalRuns(binaryImage, centerX);

    bool hasHorizontalPattern = false;
    int hCenterRunIndex = -1;
    for (int i = 0; i <= (int)horizontalRuns.size() - 5; i++) {
        if (checkPatternRatio(horizontalRuns, i, debugImage, centerY, "H")) {
            int runSum = 0;
            for (int j = 0; j < i; j++) {
                runSum += horizontalRuns[j];
            }

            int centerOfPatternX = runSum + horizontalRuns[i] + horizontalRuns[i+1] + (horizontalRuns[i+2] / 2);
            if (abs(centerX - centerOfPatternX) < (horizontalRuns[i+2] / 2.0f + 0.5f)) {
                hasHorizontalPattern = true;
                hCenterRunIndex = i+2;

                // Draw horizontal pattern indicator
                line(debugImage, Point(runSum, centerY),
                     Point(runSum + horizontalRuns[i] + horizontalRuns[i+1] + horizontalRuns[i+2] + horizontalRuns[i+3] + horizontalRuns[i+4], centerY),
                     Scalar(0, 255, 255), 2);
                break;
            }
        }
    }

    bool hasVerticalPattern = false;
    int vCenterRunIndex = -1;
    for (int i = 0; i <= (int)verticalRuns.size() - 5; i++) {
        if (checkPatternRatio(verticalRuns, i, debugImage, centerX, "V")) {
            int runSum = 0;
            for (int j = 0; j < i; j++) {
                runSum += verticalRuns[j];
            }

            int centerOfPatternY = runSum + verticalRuns[i] + verticalRuns[i+1] + (verticalRuns[i+2] / 2);
            if (abs(centerY - centerOfPatternY) < (verticalRuns[i+2] / 2.0f + 0.5f)) {
                hasVerticalPattern = true;
                vCenterRunIndex = i+2;

                // Draw vertical pattern indicator
                line(debugImage, Point(centerX, runSum),
                     Point(centerX, runSum + verticalRuns[i] + verticalRuns[i+1] + verticalRuns[i+2] + verticalRuns[i+3] + verticalRuns[i+4]),
                     Scalar(255, 255, 0), 2);
                break;
            }
        }
    }

    if (hasHorizontalPattern && hasVerticalPattern && hCenterRunIndex != -1 && vCenterRunIndex != -1) {
        float hModuleSize = (float)horizontalRuns[hCenterRunIndex] / 3.0f;
        float vModuleSize = (float)verticalRuns[vCenterRunIndex] / 3.0f;

        string moduleSizeText = "H:" + to_string((int)(hModuleSize * 10) / 10.0f) +
                               " V:" + to_string((int)(vModuleSize * 10) / 10.0f);
        putText(debugImage, moduleSizeText, Point(centerX + 10, centerY + 20),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

        if (abs(hModuleSize - vModuleSize) / max(hModuleSize, vModuleSize) > tolerance) {
            putText(debugImage, "X", Point(centerX + 5, centerY - 5),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            return false;
        }

        circle(debugImage, Point(centerX, centerY), 8, Scalar(0, 255, 0), 3);
        putText(debugImage, "FP", Point(centerX + 10, centerY - 10),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        return true;
    }

    putText(debugImage, "X", Point(centerX + 5, centerY - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
    return false;
}

struct FinderPatternCandidate {
    Point center;
    float score;
    float patternModuleSize;

    bool operator>(const FinderPatternCandidate& other) const {
        return score > other.score;
    }
};

std::vector<Point> locateFinderPatterns(const Mat& binaryImage, float& moduleSize, int& version, Mat& debugImage) {
    std::vector<FinderPatternCandidate> candidates;

    cvtColor(binaryImage, debugImage, COLOR_GRAY2BGR);

    putText(debugImage, "Scanning for Finder Patterns...", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    int scanLineCount = 0;
    int candidateCount = 0;

    for (int y = 0; y < binaryImage.rows; y += 2) {
        scanLineCount++;

        if (scanLineCount % 50 == 0) {
            line(debugImage, Point(0, y), Point(debugImage.cols, y), Scalar(100, 100, 100), 1);
            putText(debugImage, to_string(scanLineCount), Point(5, y),
                    FONT_HERSHEY_SIMPLEX, 0.3, Scalar(150, 150, 150), 1);
        }

        vector<int> runLengths = getHorizontalRuns(binaryImage, y);

        for (int i = 0; i <= (int)runLengths.size() - 5; i++) {
            if (checkPatternRatio(runLengths, i, debugImage, y, "H")) {
                candidateCount++;

                int totalWidthBeforePattern = 0;
                for (int j = 0; j < i; j++) {
                    totalWidthBeforePattern += runLengths[j];
                }
                int centerX = totalWidthBeforePattern + runLengths[i] + runLengths[i+1] + (runLengths[i+2] / 2);

                circle(debugImage, Point(centerX, y), 2, Scalar(255, 255, 0), 1);
                putText(debugImage, to_string(candidateCount), Point(centerX + 3, y - 3),
                        FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 0), 1);

                Point refinedP = refineCenter(binaryImage, centerX, y, runLengths[i] + runLengths[i+1] + runLengths[i+2] + runLengths[i+3] + runLengths[i+4]);
                centerX = refinedP.x;
                int centerY = refinedP.y;

                if (isFinderPattern(binaryImage, centerX, centerY, debugImage)) {
                    float currentPatternModuleSize = (float)(runLengths[i] + runLengths[i+1] + runLengths[i+2] + runLengths[i+3] + runLengths[i+4]) / 7.0f;
                    candidates.push_back({Point(centerX, centerY), 1.0f, currentPatternModuleSize});

                    putText(debugImage, "VALID", Point(centerX + 15, centerY + 5),
                            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
                }
            }
        }
    }

    putText(debugImage, "Scan Lines: " + to_string(scanLineCount), Point(10, 60),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(debugImage, "Candidates: " + to_string(candidateCount), Point(10, 90),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(debugImage, "Valid Patterns: " + to_string(candidates.size()), Point(10, 120),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

    std::sort(candidates.begin(), candidates.end(), [](const FinderPatternCandidate& a, const FinderPatternCandidate& b) {
        return a.score > b.score;
    });

    std::vector<Point> uniqueFinderPatterns;
    std::vector<float> uniqueFinderPatternSizes;

    int suppressedCount = 0;
    for (const auto& candidate : candidates) {
        bool isDuplicate = false;
        for (const auto& existingPattern : uniqueFinderPatterns) {
            float minDistance = candidate.patternModuleSize * 5.0f;
            int distSq = (candidate.center.x - existingPattern.x) * (candidate.center.x - existingPattern.x) +
                         (candidate.center.y - existingPattern.y) * (candidate.center.y - existingPattern.y);
            if (distSq < minDistance * minDistance) {
                isDuplicate = true;
                suppressedCount++;

                circle(debugImage, candidate.center, 5, Scalar(128, 128, 128), 1);
                putText(debugImage, "SUP", Point(candidate.center.x + 8, candidate.center.y - 8),
                        FONT_HERSHEY_SIMPLEX, 0.3, Scalar(128, 128, 128), 1);
                break;
            }
        }

        if (!isDuplicate) {
            uniqueFinderPatterns.push_back(candidate.center);
            uniqueFinderPatternSizes.push_back(candidate.patternModuleSize);

            circle(debugImage, candidate.center, 12, Scalar(0, 0, 255), 2);
            putText(debugImage, to_string(uniqueFinderPatterns.size()),
                    Point(candidate.center.x - 5, candidate.center.y + 5),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);

            cout << "Found unique finder pattern at (" << candidate.center.x << ", " << candidate.center.y << ")" << endl;
        }
    }

    putText(debugImage, "Suppressed: " + to_string(suppressedCount), Point(10, 150),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(128, 128, 128), 2);
    putText(debugImage, "Unique: " + to_string(uniqueFinderPatterns.size()), Point(10, 180),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);

    moduleSize = 0;
    if (!uniqueFinderPatternSizes.empty()) {
        float totalModuleSize = 0;
        for (float size : uniqueFinderPatternSizes) {
            totalModuleSize += size;
        }
        moduleSize = totalModuleSize / uniqueFinderPatternSizes.size();
    }

    putText(debugImage, "Avg Module Size: " + to_string((int)(moduleSize * 100) / 100.0f),
            Point(10, 210), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);

    version = -1;
    if (uniqueFinderPatterns.size() >= 3) {
        std::vector<float> distances;
        for (size_t i = 0; i < uniqueFinderPatterns.size(); i++) {
            for (size_t j = i + 1; j < uniqueFinderPatterns.size(); j++) {
                float dx = uniqueFinderPatterns[i].x - uniqueFinderPatterns[j].x;
                float dy = uniqueFinderPatterns[i].y - uniqueFinderPatterns[j].y;
                float dist = sqrt(dx*dx + dy*dy);
                distances.push_back(dist);

                line(debugImage, uniqueFinderPatterns[i], uniqueFinderPatterns[j], Scalar(255, 0, 255), 1);
                Point midpoint((uniqueFinderPatterns[i].x + uniqueFinderPatterns[j].x) / 2,
                              (uniqueFinderPatterns[i].y + uniqueFinderPatterns[j].y) / 2);
                putText(debugImage, to_string((int)dist), midpoint,
                        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
            }
        }

        if (!distances.empty()) {
            std::sort(distances.begin(), distances.end());
            float qrCodeSize = distances.back();

            vector<Point> sortedPatterns = uniqueFinderPatterns;
            sort(sortedPatterns.begin(), sortedPatterns.end(),
                 [](const Point& a, const Point& b) {
                     return (a.x + a.y) < (b.x + b.y);
                 });

            Point tl = sortedPatterns[0];
            Point temp1 = sortedPatterns[1];
            Point temp2 = sortedPatterns[2];

            Point tr, bl;
            if (temp1.x > temp2.x) {
                tr = temp1;
                bl = temp2;
            } else {
                tr = temp2;
                bl = temp1;
            }

            float distTL_TR = sqrt(pow(tl.x - tr.x, 2) + pow(tl.y - tr.y, 2));
            float distTL_BL = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
            float estimatedSideLength = (distTL_TR + distTL_BL) / 2.0f;

            putText(debugImage, "TL-TR: " + to_string((int)distTL_TR), Point(10, 240),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            putText(debugImage, "TL-BL: " + to_string((int)distTL_BL), Point(10, 260),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            putText(debugImage, "Avg Side: " + to_string((int)estimatedSideLength), Point(10, 280),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);

            if (estimatedSideLength > 0 && moduleSize > 0) {
                float estimatedTotalModules = (estimatedSideLength / moduleSize) + 7.0f;
                version = static_cast<int>(round((estimatedTotalModules - 21.0f) / 4.0f) + 1.0f);
                version = max(1, min(40, version));

                // Display version calculation
                putText(debugImage, "Est Modules: " + to_string((int)estimatedTotalModules),
                        Point(10, 300), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }
        }
    }

    if (version > 0) {
        putText(debugImage, "VERSION: " + to_string(version), Point(10, 330),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
    } else {
        putText(debugImage, "VERSION: Unknown", Point(10, 330),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    }

    cout << "Found " << uniqueFinderPatterns.size() << " unique finder patterns, average module size: "
         << moduleSize << ", estimated version: " << version << endl;

    return uniqueFinderPatterns;
}

void findQRCodePatterns(Mat src) {
    Mat graySrc;
    if (src.channels() != 1) {
        cvtColor(src, graySrc, COLOR_BGR2GRAY);
    } else {
        graySrc = src.clone();
    }

    Mat originalDisplay;
    if (src.channels() == 1)
        cvtColor(src, originalDisplay, COLOR_GRAY2BGR);
    else
        originalDisplay = src.clone();

    putText(originalDisplay, "STEP 1: Original Image", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    putText(originalDisplay, "Size: " + to_string(src.cols) + "x" + to_string(src.rows),
            Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    putText(originalDisplay, "Channels: " + to_string(src.channels()),
            Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    Mat thresholdDebug;
    Mat binary = dynamicThresholdBinarization(graySrc, thresholdDebug);

    putText(thresholdDebug, "STEP 2: Dynamic Thresholding", Point(10, thresholdDebug.rows - 100),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    Mat binaryDisplay;
    cvtColor(binary, binaryDisplay, COLOR_GRAY2BGR);
    putText(binaryDisplay, "STEP 3: Binary Result", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

    int blackPixels = 0, whitePixels = 0;
    for (int i = 0; i < binary.rows; i++) {
        for (int j = 0; j < binary.cols; j++) {
            if (binary.at<uchar>(i, j) == 0) blackPixels++;
            else whitePixels++;
        }
    }

    putText(binaryDisplay, "Black pixels: " + to_string(blackPixels),
            Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(binaryDisplay, "White pixels: " + to_string(whitePixels),
            Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    // locate finder patterns with comprehensive debugging
    int version;
    float moduleSize;
    Mat patternDebug;
    std::vector<Point> finderPatterns = locateFinderPatterns(binary, moduleSize, version, patternDebug);

    putText(patternDebug, "STEP 4: Finder Pattern Detection", Point(10, patternDebug.rows - 20),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

    Mat finalDisplay;
    if (src.channels() == 1)
        cvtColor(src, finalDisplay, COLOR_GRAY2BGR);
    else
        finalDisplay = src.clone();

    putText(finalDisplay, "STEP 5: Final Results", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

    // draw comprehensive results on final image
    for (size_t i = 0; i < finderPatterns.size(); i++) {
        const Point& center = finderPatterns[i];
        int patternDrawingSize = max(10, (int)(moduleSize * 7));

        // Draw multiple visual indicators
        circle(finalDisplay, center, patternDrawingSize/2, Scalar(0, 0, 255), 3);
        circle(finalDisplay, center, patternDrawingSize/4, Scalar(255, 0, 0), 2);
        circle(finalDisplay, center, 3, Scalar(0, 255, 0), -1);

        // Draw crosshair
        line(finalDisplay, Point(center.x - 15, center.y), Point(center.x + 15, center.y), Scalar(0, 255, 0), 3);
        line(finalDisplay, Point(center.x, center.y - 15), Point(center.x, center.y + 15), Scalar(0, 255, 0), 3);

        // Comprehensive labeling
        putText(finalDisplay, "FP" + to_string(i+1), Point(center.x + 20, center.y - 20),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        putText(finalDisplay, "(" + to_string(center.x) + "," + to_string(center.y) + ")",
                Point(center.x + 20, center.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    // draw bounding box and measurements if we have 3+ patterns
    if (finderPatterns.size() >= 3) {
        vector<Point> sortedPatterns = finderPatterns;
        sort(sortedPatterns.begin(), sortedPatterns.end(),
             [](const Point& a, const Point& b) {
                 return (a.x + a.y) < (b.x + b.y);
             });

        Point tl = sortedPatterns[0];
        Point temp1 = sortedPatterns[1];
        Point temp2 = sortedPatterns[2];

        Point tr, bl;
        if (temp1.x > temp2.x) {
            tr = temp1;
            bl = temp2;
        } else {
            tr = temp2;
            bl = temp1;
        }

        // Draw labeled bounding box
        line(finalDisplay, tl, tr, Scalar(255, 255, 0), 3);
        line(finalDisplay, tl, bl, Scalar(255, 255, 0), 3);

        Point br(bl.x + (tr.x - tl.x), tr.y + (bl.y - tl.y));
        line(finalDisplay, tr, br, Scalar(255, 255, 0), 3);
        line(finalDisplay, bl, br, Scalar(255, 255, 0), 3);

        // Label corners
        putText(finalDisplay, "TL", Point(tl.x - 30, tl.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        putText(finalDisplay, "TR", Point(tr.x + 10, tr.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        putText(finalDisplay, "BL", Point(bl.x - 30, bl.y + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        putText(finalDisplay, "BR", Point(br.x + 10, br.y + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
    }

    int yPos = 70;
    putText(finalDisplay, "Finder Patterns: " + to_string(finderPatterns.size()),
            Point(10, yPos), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    yPos += 30;

    putText(finalDisplay, "Module Size: " + to_string((int)(moduleSize * 100) / 100.0f),
            Point(10, yPos), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    yPos += 30;

    putText(finalDisplay, "QR Version: " + (version > 0 ? to_string(version) : "Unknown"),
            Point(10, yPos), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    yPos += 30;

    if (version > 0) {
        int expectedSize = 21 + (version - 1) * 4;
        putText(finalDisplay, "Expected Size: " + to_string(expectedSize) + "x" + to_string(expectedSize),
                Point(10, yPos), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    }


    imshow("Step 1: Original", originalDisplay);
  //  imshow("Step 2: Threshold Debug", thresholdDebug);
    imshow("Step 3: Binary Result", binaryDisplay);
    imshow("Step 4: Pattern Detection Debug", patternDebug);
    imshow("Step 5: Final QR Detection", finalDisplay);


    cout << "\n=== QR CODE DETECTION COMPLETE ===" << endl;
    cout << "Image size: " << src.cols << "x" << src.rows << endl;
    cout << "Found " << finderPatterns.size() << " finder patterns" << endl;
    cout << "Estimated module size: " << moduleSize << " pixels" << endl;
    cout << "Estimated QR version: " << (version > 0 ? to_string(version) : "Unknown") << endl;

    if (version > 0) {
        int expectedSize = 21 + (version - 1) * 4;
        cout << "Expected QR size: " << expectedSize << "x" << expectedSize << " modules" << endl;
        cout << "Expected image size: " << (int)(expectedSize * moduleSize) << "x" << (int)(expectedSize * moduleSize) << " pixels" << endl;
    }

    cout << "Black pixels: " << blackPixels << " (" << (100.0 * blackPixels / (src.rows * src.cols)) << "%)" << endl;
    cout << "White pixels: " << whitePixels << " (" << (100.0 * whitePixels / (src.rows * src.cols)) << "%)" << endl;

    for (size_t i = 0; i < finderPatterns.size(); i++) {
        cout << "Finder Pattern " << (i+1) << ": (" << finderPatterns[i].x << ", " << finderPatterns[i].y << ")" << endl;
    }
    cout << "=====================================" << endl;

    waitKey(0);
    destroyAllWindows();
}


int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

    // Get current directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        projectPath = std::string(cwd);
    }

    int op;
    do {
        system("clear");
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - Open image\n");
        printf(" 2 - Negative image\n");
        printf(" 3 - Additive factor\n");
        printf(" 4 - Multiplicative factor\n");
        printf(" 5 - 4 color Image\n");
        printf(" 6 - Inverse matrix\n");
        printf(" 7 - Mirror img\n");
        printf(" 8 - Mirror img 2\n");
        printf(" 9 - Mirror img color\n");
        printf(" 10 - Mirror img color 2 \n");
        printf(" 11 - snap image \n");
        printf(" 12 - snap video inverted \n");
        printf(" 12 - snap video inverted \n");
        printf(" 13 - img to 3 channel \n");
        printf(" 14 - color to grayscale \n");
        printf(" 15 - color to grayscale 2\n");
        printf(" 16 - binarizare 2\n");
        printf(" 17` - rgb to hsv 2\n");
        printf(" 18` - make and display histogram 2\n");
        printf(" 19` - make and display fdp histogram 2\n");
        printf(" 20` - reduced cols histogram\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op) {
            case 1:
                openImageTest();
            break;
            case 2:
                negativeImage();
            break;
            case 3:
                additiveFactor();
            break;
            case 4:
                multipFactor();
            break;
            case 5:
                colorImage();
            break;
            case 6:
                printInverseMatrix();
            break;
            case 7:
                mirrorImage();
            break;
            case 8:
                mirrorImage2();
            break;
            case 9:
                mirrorImageColor();
            break;
            case 10:
                mirrorImageColor2();
            break;
            case 11:
                testSnap();
            break;
            case 12:
                testSnap2();
            break;
            case 13:
                convColorTo3GrayScale();
            break;
            case 14:
                convColorToGrayScale();
            break;
            case 15:
                convColorToGrayScale2();
            break;
            case 16:
                convDstToBinary();
            break;
            case 17:
                convRGBToHSV();
            break;
            case 18:{
                int histogram[256]={};
                computeHistogram("./Images/kids.bmp",histogram,256);
                showHistogram("img",histogram,256,256);
                break;
            }
            case 19:{
                float histogram[256]={};
                computeHistogram("./Images/kids.bmp",histogram,256);
                showHistogram("img",histogram,256,256);
                break;
            }
            case 20: {
                int histogram[256] = {};
                int m;
                scanf("%d", &m);
                computeHistogram("./Images/kids.bmp", histogram, m);
                showHistogram("./Images/kids.bmp", histogram, 256, 256);
                break;
            }
            case 21: {
                geometry("./Images/oval_obl.bmp");
                break;
            }
             case 22: {
                labelMatrix("./Images/crosses.bmp");
                break;
            }
            case 23: {
                Mat src = imread("./Images/wdg2thr3_bw.bmp", IMREAD_GRAYSCALE);
                Mat dst = dilatare(src);
                imshow("orig", src);
                imshow("dilatata", dst);
                waitKey(0);
                break;
            }
            case 24: {
                Mat src = imread("./Images/reg1neg1_bw.bmp", IMREAD_GRAYSCALE);
                //Mat dst = dilatare(src);
                Mat dst = eroziune(src);
                imshow("orig", src);
                imshow("dilatata", dst);
                waitKey(0);
                break;
            }
             case 25: {
                Mat src = imread("./Images/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
                //Mat dst = dilatare(src);
                Mat dst = eroziune(src);
                Mat dst2 = dilatare(dst);
                imshow("orig", src);
                imshow("open", dst);
                waitKey(0);
                break;
            }
             case 26: {
                Mat src = imread("./Images/art4_bw.bmp", IMREAD_GRAYSCALE);
                //Mat dst = dilatare(src);
                Mat dst = dilatare(src);
                Mat dst2 = eroziune(dst);
                imshow("orig", src);
                imshow("close", dst);
                waitKey(0);
                break;
            }
            case 27: {
                Mat src = imread("./Images/wdg2thr3_bw.bmp", IMREAD_GRAYSCALE);
                Mat dst = extractContour(src);
            }
            case 28: {
                Mat src = imread("./Images/reg1neg1_bw.bmp", IMREAD_GRAYSCALE);
                Mat dst = fillObject(src);
            }
case 29: {
    Mat src = imread("./Images/Project/qr.jpg", IMREAD_GRAYSCALE);
    Mat src1 = imread("./Images/Project/Q1R.jpg", IMREAD_GRAYSCALE);
    Mat src3 = imread("./Images/Project/iphoneqr.jpg", IMREAD_GRAYSCALE);
    Mat src2 = imread("./Images/Project/qr-wifi-card.jpg", IMREAD_GRAYSCALE);
 //   findQRCodePatterns(src3);

    findQRCodePatterns(src1);
    findQRCodePatterns(src);
    findQRCodePatterns(src2);
    break;
}
case 30: {
    Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
  //  dynamicThresholdBinarization(src);
    break;
}
            case 31: {
    Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
    medium_intensity(src);
    break;
}
                   case 32: {
    Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
    contrast(src,50,100);
    break;
}
            case 33: {
    Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
    equalizeHistogram(src);
    break;
}
            case 34: {
                Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
                int nucleu1[3][3] = {{1,1,1},{1,1,1},{1,1,1}};
                int nucleu2[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
                convolTreceJos(src,nucleu2);
                break;
            }
               case 35: {
                Mat src = imread("./Images/cameraman.bmp", IMREAD_GRAYSCALE);
                int nucleu1[3][3] = {{1,1,1},{1,1,1},{1,1,1}};
                int nucleu2[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
                int nucleu3[3][3] = {{0,-1,0},{-1,4,-1},{0,-1,0}};
                convolTreceSus(src,nucleu3);
                break;
            }
              case 36: {
                Mat src = imread("./Images/lab10/balloons_Salt&Pepper.bmp", IMREAD_GRAYSCALE);
                spatialFilters(src);
                break;
            }
             case 37: {
                Mat src = imread("./Images/lab10/balloons_Gauss.bmp", IMREAD_GRAYSCALE);
                gauss(src);
                break;
            }
        }

        } while (op!=0);
    return 0;
}