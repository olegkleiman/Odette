//
//  SSDProcessor.hpp
//  odette
//
//  Created by Oleg Kleiman on 10/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#ifndef SSDProcessor_hpp
#define SSDProcessor_hpp

#include <iostream>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"

#include "IProcessor.hpp"

using namespace std;
using namespace cv;

class SSDProcessor : public IProcessor {
    
    private:
        dnn::Net       m_net;
        vector<String> m_layerOutputNames;
        float          m_confidenceThreshold;
        static string  m_CLASSES[];
        float          m_nmsThreshold;
    
        void postprocess(Mat& frame, const std::vector<Mat>& outs);
        void postprocess(Mat& frame, Mat& detection);
        void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    
    public:
        SSDProcessor(string pathDescription, string pathModel, float confidenceThreshold);
        void process(Mat frame, int nCurrentFrame,
                     bool withMultipleScale = false,
                     bool withDraw = true);

};



#endif /* SSDProcessor_hpp */
