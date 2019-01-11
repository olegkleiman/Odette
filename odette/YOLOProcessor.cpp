//
//  YOLOProcessor.cpp
//  odette
//
//  Created by Oleg Kleiman on 10/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "YOLOProcessor.hpp"

using namespace std;
using namespace cv;

YOLOProcessor::YOLOProcessor(string pathConfig,
                             string pathModel,
                             string classNamesPath,
                             float confidenceThreshold) {
    m_net = dnn::readNet(pathModel, pathConfig);
//    m_net = dnn::readNetFromDarknet(pathConfig, pathModel);
    m_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    
    ifstream classNamesFile(classNamesPath);
    if (classNamesFile.is_open()) {
        string className = "";
        while( getline(classNamesFile, className) )
            m_classNamesVec.push_back(className);
    }
    
    m_confidenceThreshold = confidenceThreshold;
    m_nmsThreshold = 0.4;
}

void YOLOProcessor::process(Mat frame, int nCurrentFrame,
                           bool withMultipleScal,
                           bool withDraw) {
    // Create a 4D blob from a frame.
//    Mat inputBlob = dnn::blobFromImage(frame, 0.007843, Size(300, 300), Scalar(), true, false);
    // For sizes & other params,
    // See https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection
    Mat inputBlob;
    double scaleFactor = 1/255.;
    dnn::blobFromImage(frame, inputBlob,
                       scaleFactor,
                       Size(416, 416), // size WxH
                       Scalar(0,0,0)); // Mean subtraction
    m_net.setInput(inputBlob);
    
    vector<Mat> outs;
    m_net.forward(outs, getOutputsNames(m_net));

    postprocess(frame, outs);

//    Mat detectionMat = m_net.forward();//outs[2]);
//    for (int i = 0; i < detectionMat.rows; i++) {
//        const int probability_index = 5;
//        const int probability_size = detectionMat.cols - probability_index;
//        float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
//        size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
//        float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
//        if (confidence > m_confidenceThreshold) {
//            float x_center = detectionMat.at<float>(i, 0) * frame.cols;
//            float y_center = detectionMat.at<float>(i, 1) * frame.rows;
//            float width = detectionMat.at<float>(i, 2) * frame.cols;
//            float height = detectionMat.at<float>(i, 3) * frame.rows;
//            Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
//            Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
//            Rect object(p1, p2);
//
////            Scalar object_roi_color(0, 255, 0);
////            if (object_roi_style == "box")
////            {
////                rectangle(frame, object, object_roi_color);
////            }
////            else
////            {
////                Point p_center(cvRound(x_center), cvRound(y_center));
////                line(frame, object.tl(), p_center, object_roi_color, 1);
////            }
////            String className = objectClass < m_classNamesVec.size() ? m_classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
////            String label = format("%s: %.2f", className.c_str(), confidence);
////            int baseLine = 0;
////            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
////            rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)),
////                      object_roi_color, FILLED);
////            putText(frame, label, p1 + Point(0, labelSize.height),
////                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
//        }
//    }
}

void YOLOProcessor::postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > m_confidenceThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, m_confidenceThreshold, m_nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void YOLOProcessor::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!m_classNamesVec.empty())
    {
        CV_Assert(classId < (int)m_classNamesVec.size());
        label = m_classNamesVec[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
}

// Get the names of the output layers
vector<string> YOLOProcessor::getOutputsNames(const dnn::Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
