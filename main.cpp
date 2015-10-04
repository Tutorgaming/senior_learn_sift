/************************************************
Name : CV_Playground
Version : 3
Detail : SIFT Algorithm with KNN Implemented
Author : Theppasith N.
28-AUG-2015 08:00AM
************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann.hpp>

using namespace std;
using namespace cv;

Mat capturedImage;

//int captureImg(){
//    VideoCapture camera(0);
//    char x;
//    while(1){
//        x = waitKey(10);
//        camera >> capturedImage;
//        imshow("capture",capturedImage);
//        if( x=='s'){
//            return 0;
//        }
//    }
//}

int main(){
    //Object Image
    Mat inputImage,vidStream;                             //Input Image
    inputImage = imread("loft.jpg");            //Load Image
    ////////////////////////////////////////
    //captureImg();
    //Convert To Gray => to SIFT
    if(inputImage.empty()){
        std::cout << "No Such File" <<std::endl;
        return -1;
    }
        Mat inputGray;
        cv::cvtColor(inputImage , inputGray , CV_BGR2GRAY);
    //SOURCE Object SIFT Keypoint Detector
        Mat SelectedArea;

        //cv::xfeatures2d::SiftFeatureDetector detector;
        std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(0,4,0.04,10,1.6);
        sift->detect(inputImage,keypoints); //inputGrey

    //SOURCE Object SIFT Descriptor
        //cv::xfeatures2d::SiftDescriptorExtractor extractor;
        cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor = cv::xfeatures2d::SIFT::create(3.0);
        Mat descriptor;
        extractor->compute(inputImage,keypoints,descriptor);//inputGrey

    //Draw Keypoints on Object Image
        //Mat sourceSift;
        //drawKeypoints(inputImage,keypoints,sourceSift);

    //Camera Set-up
        VideoCapture cap(0);
        if(!cap.isOpened())return -1;
    //Process Video input
        Mat videoInput,videoOutput;
        Mat videoGray;
    //Detection on videoInput
        //cv::xfeatures2d::SiftFeatureDetector detector_vdo;
        cv::Ptr<cv::xfeatures2d::SIFT> sift_vdo = cv::xfeatures2d::SIFT::create(0,4,0.04,40,1.6);
        cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor_vdo = cv::xfeatures2d::SIFT::create(3.0);
        Mat descriptor_vdo;
        vector<cv::KeyPoint> keypoints_vdo;
    //Matcher
        cv::FlannBasedMatcher matcher;//cv::BFMatcher matcher;
        //vector<vector<cv::DMatch> > matches;
        //cout << keypoints.size() <<endl;

        float thresholdMatchingNN=0.7;
       //unsigned int thresholdGoodMatches=10;


        while(true){
            vector<DMatch > good_matches; // Match on KNN
            vector<vector<cv::DMatch> > matches; //Match On Bipartite
            //Receive Frames From Camera
                cap >> videoInput;
            //Convert To GreyScale for SIFT
                cv::cvtColor(videoInput,videoGray,CV_BGR2GRAY);
            //Create the SIFT Detector
                sift_vdo->detect(videoInput,keypoints_vdo); // Grey
                //drawKeypoints(videoInput,keypoints_vdo,videoInput);
            //Extract the Descriptor
                extractor_vdo->compute(videoGray,keypoints_vdo,descriptor_vdo);
            //Do The KNN Matches
                matcher.knnMatch(descriptor,descriptor_vdo,matches,2);
                for(int i = 0; i < min(descriptor_vdo.rows-1,(int) matches.size()); i++){ //THIS LOOP IS SENSITIVE TO SEGFAULTS
                    if((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance))
                        && ((int) matches[i].size()<=2 && (int) matches[i].size()>0)){
                        good_matches.push_back(matches[i][0]);
                    }
                }
            //Originally
            /*
             matcher.match(descriptor,descriptor_vdo,matches);
             drawMatches(inputImage,keypoints,videoInput,keypoints_vdo,matches,videoOutput,Scalar::all(-1),Scalar::all(-1),vector<vector<char> >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            */
            //Draw only "good" matches from NN
                drawMatches( inputImage, keypoints, videoInput, keypoints_vdo, good_matches, videoOutput, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            //Show Output
                imshow("Video Stream",videoOutput);     //Display Video Stream output
            //Press X to Quit
                char x = waitKey(1); // wait forever for an user key press.
                if(x == 'q'){
                        std::cout << "Q Detected => Quit" <<endl;
                        break;
                }
        }


    return 0;
}
