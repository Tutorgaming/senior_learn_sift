/************************************************
Name : CV_Playground
Version : 3-4
Detail : SIFT Algorithm with KNN Implemented
Author : Theppasith N.
28-AUG-2015 08:00AM
4-OCT-2015 02:00AM
REF :
http://stackoverflow.com/questions/22722772/how-to-use-sift-in-opencv
http://stackoverflow.com/questions/17967950/improve-matching-of-feature-points-with-opencv
http://stackoverflow.com/questions/23440694/heap-corruption-exception-in-knnmatch-opencv
http://morf.lv/modules.php?name=tutorials&lasit=2
http://www.programering.com/a/MTNzAzMwATc.html
REF : (Mouse Event)
https://jayrambhia.wordpress.com/2012/09/20/roi-bounding-box-selection-of-mat-images-in-opencv/
************************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann.hpp>

using namespace std;
using namespace cv;
cv::Point firstClick,secondClick;
Mat capturedImage,capturedSelection;
Rect selection;
bool dragflag = false;
bool finished = false;

void myMouseEvent(int event , int x ,int y , int flags , void* param ){
    if(event == CV_EVENT_LBUTTONDOWN && !dragflag){
        //Left Mouse Click => Set State to clicked
        firstClick = Point(x,y);
        dragflag = true;
    }
    if(event == CV_EVENT_MOUSEMOVE && dragflag){
        //Determine the end point of rectangle and Show Drawing on screen
        Mat img1 = capturedImage.clone(); //Mock Up for Visualizing
        secondClick = Point(x,y);
        rectangle(img1,firstClick,secondClick,CV_RGB(0,255,0),3,8,0); //Draw on Visualizer
        imshow("crop",img1); //Display Visualizer instead of real mat
    }
    if(event == CV_EVENT_LBUTTONUP && dragflag){
        //Mouse Release => Get the Selected Area to CaptureSelection
        secondClick = Point(x,y);
        selection = Rect(firstClick.x,firstClick.y,abs(firstClick.x - secondClick.x),abs(firstClick.y-secondClick.y)); //Size of Selected Area
        dragflag = false;
        capturedSelection = capturedImage(selection);
    }
    if(event == CV_EVENT_LBUTTONUP){
        //State For Mouse Click but no drag
        finished = true;
        dragflag = false;
    }
}


int captureImg(){
    VideoCapture camera(0);
    namedWindow("PRESS S TO CAPTURE");
    char keyboardInput;
    while(1){
        keyboardInput = waitKey(10);
        camera.read(capturedImage);
        imshow("PRESS S TO CAPTURE",capturedImage);
        if( keyboardInput=='s'){
            destroyWindow("PRESS S TO CAPTURE");
            return 0;
        }
    }
}

int cropSelection(){
    namedWindow("crop");
    while(1){
       cv::waitKey(10);
        imshow("crop",capturedImage);
        cvSetMouseCallback("crop", myMouseEvent, NULL);
        if(finished && !capturedSelection.empty()){
            cout << "Cropping finished" <<endl;
            destroyWindow("crop");
            //namedWindow("cropped");
            //imshow("cropped",capturedSelection);
            return 0;
        }
    }
}


int main(){
    //Mat For Image
    Mat inputImage,vidStream;                     //Input Image
    int manuallySelect = 0;
    cout << "0 = Load JPG" << endl << "1 = Capture from Device" <<endl << " = " ;
    cin >> manuallySelect ;
    //Load Image
    if(!manuallySelect){
        inputImage = imread("loft.jpg");
    }else{
        captureImg();
        cropSelection();
        inputImage = capturedSelection;
    }
    //Convert To Gray => to SIFT
    if(inputImage.empty()){
        std::cout << "No Such INPUT" <<std::endl;
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

        float thresholdMatchingNN=0.8;
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
                //cout << "Amount is = " << good_matches.size() <<endl;
                if(good_matches.size() > 120){
                    cout <<"@@@@@@[" << good_matches.size() << "]Detected Object" <<endl;
                }else{
                    cout <<"[" << good_matches.size() << "]No Detected" <<endl;
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
