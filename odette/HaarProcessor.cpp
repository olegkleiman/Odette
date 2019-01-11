//
//  HaarProcessor.cpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <iostream>

#include "IProcessor.hpp"
#include "HaarProcessor.hpp"
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

HaarProcessor::HaarProcessor(string cascadeName,
                             bool withTrace) {
    
    string pathToCascasde = "./cascades/" + cascadeName;
    
    m_boxes_cascade.load(pathToCascasde);
    if(m_boxes_cascade.empty() ) {
        cout << "Failed to load cascade" << endl;
    }
};

void HaarProcessor::process(Mat frame, int nCurrentFrame,
                            bool withMultipleScale,
                            bool withDraw) {

    if( withMultipleScale )
        m_boxes_cascade.detectMultiScale(frame, m_boxes);//, 1.1, 2); //, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
//    else
//        m_boxes_cascade.detect(frame, m_boxes);

    if( withDraw )
        draw(frame);
}

void HaarProcessor::draw(Mat frame) {
    
    for( int i = 0; i < m_boxes.size(); i++ ) {
        rectangle(frame, m_boxes[i], Scalar(0,255,0), 2);
    }
    
}

