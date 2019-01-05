//
//  main.cpp
//  odette
//
//  Created by Oleg Kleiman on 05/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <iostream>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn/shape_utils.hpp>

using namespace std;
using namespace cv;

std::string CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"};

float confidenceThreshold = 0.2;

void on_trackbar(int value, void *userData) {
    cout << value << endl;
    confidenceThreshold = value/10.;
}

int main(int argc, const char * argv[]) {
    
    dnn::Net caffeNet;
    
    try {
        //    Edit scheme -> Options -> Working directory
        caffeNet = dnn::readNetFromCaffe("./models/MobileNet/MobileNetSSD_deploy.prototxt.txt",
                                         "./models/MobileNet/MobileNetSSD_deploy.caffemodel");
    } catch(Exception ex) {
        cout << ex.msg << endl;
        return 1;
    }
    
    VideoCapture capture("./data/i11.mov");
    if(!capture.isOpened()){
        std::cout<<"cannot read video!\n";
        return -1;
    }
    
    // Setup OpenCV GUI
    string winName = "SSD/MobileNet", trackbarName = "Confidence Threshold";;
    namedWindow(winName, WINDOW_AUTOSIZE);
    int confidenceThresholdValue = 2;
    createTrackbar(trackbarName, winName, &confidenceThresholdValue, 10, on_trackbar);
    on_trackbar(confidenceThresholdValue, 0);
    
    Mat frame;
    
    bool odd = true;
    while( true ) {
        
        capture >> frame;

        if( frame.empty() ){
            cout << "Skip the empty frame" << endl;
            return 1;
        }
        resize(frame, frame, Size(frame.cols/2, frame.rows/2));

        // Construct an input blob for the image
        // by resizing to a fixed 300x300 pixels and then normalizing it
        // (note: normalization is done via the authors of the MobileNet SSD
        // implementation)
        Mat inputBlob = dnn::blobFromImage(frame, 0.007843, Size(300, 300), Scalar(), true, false);
        caffeNet.setInput(inputBlob);
        Mat detection = caffeNet.forward();
        
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
        ostringstream ss;
        
        for (int i = 0; i < detectionMat.rows; i++) {
            
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidenceThreshold) {
                
                int idx = static_cast<int>(detectionMat.at<float>(i, 1));
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                
                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
                
                rectangle(frame, object, Scalar(0, 255, 0), 2);
                
                cout << CLASSES[idx] << ": " << confidence << endl;
                
                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = CLASSES[idx] + ": " + conf;
//                int baseLine = 0;
//                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                putText(frame, label,
                        Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
            }
            
        }

        vector<double> layersTimings;
        double tick_freq = getTickFrequency();
        double time_ms = caffeNet.getPerfProfile(layersTimings) / tick_freq * 1000;
        putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f / time_ms, time_ms),
                Point(20, 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        if( odd) {
            imshow(winName, frame);
        }
        odd = !odd;
        
        if( waitKey(1) >=0 )
            break;
    }
    
    return 0;
}
