//
//  HOGMethod.cpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "HOGProcessor.hpp"

using namespace std;
using namespace cv;


_HOG::_HOG(bool withTrace) {
//    m_hog = HOGDescriptor(Size(48,96), Size(16,16), Size(8,8), Size(8,8), 9);//, -1.0, 0.2, 1, p_HogScaleLevels);
    m_hog = HOGDescriptor(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);
//    m_hog = HOGDescriptor(Size(48,96), Size(16,16), Size(8,8), Size(8,8), 9, 1, -1,
//                      HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
    m_hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector()); // 48x96
//    m_hog.gammaCorrection = true;
    
}

void _HOG::process(Mat frame, int nCurrentFrame,
                   bool withMultipleScale,
                   bool withDraw) {
    
    if( withMultipleScale ) {
        
        vector<Rect> found;
        vector<double> weights;
        
//        m_hog.detectMultiScale(frame, found, weights);
        m_hog.detectMultiScale(frame, found, 0, Size(8,8), Size(32,32), 1.05, 2, false);
        
        // draw detections and store location
        for( size_t i = 0; i < found.size(); i++ ) {
            
            stringstream temp;
            temp << weights[i];
            const string _str = temp.str();
            float confid = stof(_str);
            
            if( confid > 1.1 ) {
                Rect r = found[i];
                m_founds.push_back(r);
            }
        }
        
    } else {
        
        vector<cv::Point> foundLocations;
        m_hog.detect(frame, foundLocations, 0, Size(8,8));
        
        for( size_t i = 0; i < foundLocations.size(); i++ ) {
            
            Rect r =  Rect(foundLocations[i].x, foundLocations[i].y, 100, 100);
            m_founds.push_back(r);
        }
        
    }
    
    if( withDraw )
        draw(frame);

}

void _HOG::draw(Mat frame) {
    
    for(vector<Rect>::iterator it = m_founds.begin();
        it != m_founds.end();
        it++) {
        
        Rect rect = *it;
//        adjustRect(rect);
        rectangle(frame, rect, Scalar(0,255,0), 2);
        putText(frame, "found", Point(rect.x,rect.y+50),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
    }
    
    m_founds.clear();
    
}

void _HOG::adjustRect(Rect & r) const
{
    // The HOG detector returns slightly larger rectangles than the real objects,
    // so we slightly shrink the rectangles to get a nicer output.
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
}


