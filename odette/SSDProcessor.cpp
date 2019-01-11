//
//  SSDProcessor.cpp
//  odette
//
//  Created by Oleg Kleiman on 10/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <iostream>

#include "SSDProcessor.hpp"

string SSDProcessor::m_CLASSES[] = {"background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"};

SSDProcessor::SSDProcessor(string pathDescription,
                           string pathModel,
                           float confidenceThreshold) {
    //m_net = dnn::readNetFromCaffe(pathDescription, pathModel);
    
    // For dnn::readNet,  an order of @p model and @p config does not matter
    m_net = dnn::readNet(pathModel, pathDescription);
    m_net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(dnn::DNN_TARGET_OPENCL);
    
    m_confidenceThreshold = confidenceThreshold;
    m_nmsThreshold = 0.4;
    
    m_layerOutputNames = m_net.getUnconnectedOutLayersNames();
}

void SSDProcessor::process(Mat frame, int nCurrentFrame,
                           bool withMultipleScal,
                           bool withDraw) {
    // For sizes & other params,
    // See https://github.com/opencv/opencv/tree/master/samples/dnn#object-detection
    double scaleFactor = 2/255.;
    Mat inputBlob = dnn::blobFromImage(frame, scaleFactor,
                                       Size(300, 300), // size WxH
                                       Scalar(127.5, 127.5, 127.5), // Mean subtraction
                                       true, false);
    m_net.setInput(inputBlob);//, "", 0.5);
    
    vector<Mat> outs;
//    m_net.forward(outs, m_layerOutputNames);
//    postprocess(frame, outs);
    Mat detection = m_net.forward(m_layerOutputNames[0]);
    postprocess(frame, detection);
}

void SSDProcessor::postprocess(Mat& frame, Mat& detection) {
    
        MatSize detectionSize = detection.size;
        Mat detectionMat(detectionSize[2], detectionSize[3], CV_32F, detection.ptr<float>());
    
        ostringstream ss;
        for (int i = 0; i < detectionMat.rows; i++) {
    
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > m_confidenceThreshold ) {
    
                int idx = static_cast<int>(detectionMat.at<float>(i, 1));
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
    
                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));
    
                rectangle(frame, object, Scalar(0, 255, 0), 2);
    
                cout << m_CLASSES[idx] << ": " << confidence << endl;
    
                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = m_CLASSES[idx] + ": " + conf;
    //                int baseLine = 0;
    //                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                putText(frame, label,
                        Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255));
    
            }
        }
    
        vector<double> layersTimings;
        double tick_freq = getTickFrequency();
        double time_ms = m_net.getPerfProfile(layersTimings) / tick_freq * 1000;
        putText(frame, format("FPS: %.2f ; time: %.2f ms", 1000.f / time_ms, time_ms),
                Point(20, 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

}

void SSDProcessor::postprocess(Mat& frame, const std::vector<Mat>& outs)
{
    static std::vector<int> outLayers = m_net.getUnconnectedOutLayers();
    static std::string outLayerType = m_net.getLayer(outLayers[0])->type;
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > m_confidenceThreshold )
            {
                int left = (int)data[i + 3];
                int top = (int)data[i + 4];
                int right = (int)data[i + 5];
                int bottom = (int)data[i + 6];
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > m_confidenceThreshold)
            {
                int left = (int)(data[i + 3] * frame.cols);
                int top = (int)(data[i + 4] * frame.rows);
                int right = (int)(data[i + 5] * frame.cols);
                int bottom = (int)(data[i + 6] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
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
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    
    std::vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, m_confidenceThreshold, m_nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void SSDProcessor::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    
    std::string label = format("%.2f", conf);
    //if( !m_CLASSES. empty() )
    //{
//        CV_Assert(classId < (int)m_CLASSES());
        label = m_CLASSES[classId] + ": " + label;
//    }
    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

