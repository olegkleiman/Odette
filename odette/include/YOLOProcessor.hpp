//
//  YOLOProcessor.hpp
//  odette
//
//  Created by Oleg Kleiman on 10/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#ifndef YOLOProcessor_hpp
#define YOLOProcessor_hpp

#include <iostream>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "IProcessor.hpp"

using namespace std;
using namespace cv;

class YOLOProcessor : public IProcessor {

    private:
        dnn::Net m_net;
        float m_confidenceThreshold;
        float m_nmsThreshold;
        vector<String> m_classNamesVec;
        vector<string> getOutputsNames(const dnn::Net& net);
        void postprocess(Mat& frame, const vector<Mat>& outs);
        void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    
    public:
        YOLOProcessor(string pathDescription, string pathModel, string classNamesPath, float confidenceThreshold);
        void process(Mat frame, int nCurrentFrame,
                     bool withMultipleScale = false,
                     bool withDraw = true);

};

#endif /* YOLOProcessor_hpp */
