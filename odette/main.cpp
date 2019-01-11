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

#include "ProcessorFactory.hpp"
#include "IProcessor.hpp"

//#include <caffe/caffe.hpp>

using namespace std;
using namespace cv;

int currentFrameNum = 0;

string STILL_EXTENSIONS[] = { "jpg", "jpeg", "png" };

float confidenceThreshold = 0.2;
string layerOutputName;
bool volatile isStopped = false;

void on_trackbar(int value, void *userData) {
    cout << value << endl;
    confidenceThreshold = value/10.;
}

int volatile gVideoPosition = 0;
int gTotalFrames;

void on_trackbarPosition(int value, void *userData) {
    gVideoPosition = gTotalFrames / 100 * value;
}

void mouseCallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if  ( event == EVENT_LBUTTONDOWN ) {
        isStopped = !isStopped;
        cout << currentFrameNum << endl;
    }
}

static const char* params =
"{ method         | Classic/HOG/Haar/SSD/YOLO | detection method}"
"{ source         |                           | path to video file}"
"{ min_confidence | 0.24                      | min confidence      }";


#pragma mark - Main

int main(int argc, const char * argv[]) {
    
    String videoSource, detectionMethod;
    bool bStillMode = false;
    
    CommandLineParser parser(argc, argv, params);
    parser.about("Odette v1.0.0");
    
    if( parser.has("method") ) {
        detectionMethod = parser.get<String>("method");
        if( detectionMethod.empty() ) {
            parser.printMessage();
            return 0;
        }
        
        videoSource = parser.get<String>("source");
        if( videoSource.empty() ) {
            parser.printMessage();
            return 0;
        }
        
        char sep = '.';
        size_t extPoision = videoSource.rfind(sep, videoSource.length());
        if( extPoision != string::npos ) {
            string extension = videoSource.substr(extPoision+1, videoSource.length());
            string *matchStillExtension = find(begin(STILL_EXTENSIONS), end(STILL_EXTENSIONS), extension);
            if( matchStillExtension != end(STILL_EXTENSIONS) )
                bStillMode = true;
            cout << extension << endl;
        }
        
        cout << "Using " << detectionMethod << " for processing " << videoSource << endl;

        
    } else {
        parser.printMessage();
        return 0;
    }
    
    if (!cv::ocl::haveOpenCL()) {
        cout << "OpenCL is not avaiable..." << endl;
    } else {
        cv::ocl::Context context;
        if (!context.create(cv::ocl::Device::TYPE_GPU))
        {
            cout << "Failed creating the context..." << endl;
        }
        cout << context.ndevices() << " GPU devices are detected." << endl;
        for (int i = 0; i < context.ndevices(); i++)
        {
            cv::ocl::Device device = context.device(i);
            cout << "name                 : " << device.name() << endl;
            cout << "available            : " << device.available() << endl;
            cout << "imageSupport         : " << device.imageSupport() << endl;
            cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
            cout << endl;
        }
    }
    
    IProcessor *processor = ProcessorFactory::create(detectionMethod);
    
    VideoCapture capture(videoSource.c_str());
    if(!capture.isOpened()){
        std::cout<<"cannot read video!\n";
        return -1;
    }
    int frameWidth = capture.get(CAP_PROP_FRAME_WIDTH);
    cout << "Frame width: " << frameWidth;
    int frameHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
    cout << " Frame Height: " << frameHeight;
    int fps = capture.get(CAP_PROP_FPS);
    cout << " FPS: " << fps << endl;

//    double start_frame_number = 6000.0;
//    capture.set(CAP_PROP_POS_FRAMES, start_frame_number);
    
    // Setup OpenCV GUI
    string winName = "Odette", trackbarName = "Confidence Threshold", trackBarPosisitionName = "Position";
    namedWindow(winName, WINDOW_AUTOSIZE);
    int confidenceThresholdValue = 2;
    createTrackbar(trackbarName, winName, &confidenceThresholdValue, 10, on_trackbar);
    on_trackbar(confidenceThresholdValue, NULL);

    int nVideoPosition = 0;
    gTotalFrames = capture.get(CAP_PROP_FRAME_COUNT);
    createTrackbar(trackBarPosisitionName, winName, &nVideoPosition, 100, on_trackbarPosition);
    
    setMouseCallback(winName, mouseCallBackFunc, NULL);
    
    Mat frame;
    TickMeter tm;

    if( bStillMode ) {
        
        frame = imread(videoSource.c_str(), cv::IMREAD_COLOR);
        resize(frame, frame, Size(frame.cols*2, frame.rows*2));
        processor->process(frame, currentFrameNum);
        imshow(winName, frame);
        waitKey(0);
        
    } else {
    
            bool odd = true;
            while( true ) {
            
                if( isStopped ) {
                    waitKey(1);
                    continue;
                }

                capture >> frame;

                if( frame.empty() ){
                    cout << "Skip the empty frame" << endl;
                    return 1;
                }
                currentFrameNum = capture.get(CAP_PROP_POS_FRAMES);
//                resize(frame, frame, Size(frame.cols/2, frame.rows/2));

                tm.start();
                processor->process(frame, currentFrameNum);
                tm.stop();
                double time_ms = tm.getTimeMilli();
                //cout << "Processed for "  << tm.getTimeMilli() << " ms. " << endl;
                tm.reset();
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
}

    
    return 0;
}
