//
//  ClassicProcessor.cpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ClassicProcessor.hpp"

using namespace std;
using namespace cv;

Scalar RED = Scalar(0,0,255);
Scalar BLUE = Scalar(255,0,0);
Scalar GREEN = Scalar(0,255,0);
Scalar BLACK = Scalar(0,0,0);
Scalar WHITE = Scalar(255,255,255);

ClassicProcessor::ClassicProcessor(bool withTrace) {
    m_minHeight = 80;
    m_minWidth = 60;
    
    m_withTrace = withTrace;
    if( withTrace ) {
        m_winTraceName1 = "Odette: Trace 1";
        m_winTraceName2 = "Odette: Trace 2";
        namedWindow(m_winTraceName1, WINDOW_AUTOSIZE);
        namedWindow(m_winTraceName2, WINDOW_AUTOSIZE);
    }
}

void ClassicProcessor::process(Mat frame, int nCurrentFrame,
                               bool withMultipleScale,
                               bool withDraw) {

    Mat img_gray;
    cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);
    blur(img_gray, img_gray, Size(5,5));
    
    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    
    Mat img_threshold;
    threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU + THRESH_BINARY);
    
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3) );
    morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
    
    if( m_withTrace ) {
        imshow(m_winTraceName1, img_threshold);
    }

    Mat frameForCountours;
    cvtColor(frame, frameForCountours, CV_8UC3);
    
    Mat canny_output;
    int thresh = 100;
    int max_thresh = 255;
    Canny(img_gray, canny_output, thresh, thresh*2, 3 );
    
    if( m_withTrace ) {
        imshow(m_winTraceName2, canny_output);
    }
    
//    findContours(img_threshold,
    findContours(canny_output,
                 m_contours,
                 RETR_TREE, //RETR_EXTERNAL, // retrieve the external contours
                 CHAIN_APPROX_SIMPLE);
    
    if( withDraw )
        draw(frame, nCurrentFrame);
    
}

Point tb1_tl = Point(420,200);
Point tb1_br = Point(660,490);

void ClassicProcessor::draw(Mat frame, int nCurrentFrame) {
    
    string tbCaption = "trash bin";
    int shiftStartFrame = 302;
    int shiftEndFrame = 308;
    int xShift = -1;

    if( nCurrentFrame > shiftStartFrame
       && nCurrentFrame < shiftEndFrame ) {
        tb1_tl.x += xShift;
        tb1_br.x += xShift;
    }
    Rect tb1 = Rect(tb1_tl, tb1_br);
    rectangle(frame, tb1, GREEN, 2);
    drawCaption(frame, tb1, tbCaption);

    Rect tb2 = Rect(Point(500,240), Point(890,530));
    rectangle(frame, tb2, GREEN, 2);
    drawCaption(frame, tb2, tbCaption);

    Rect tb3 = Rect(Point(650,340), Point(1040,580));
    rectangle(frame, tb3, GREEN, 2);
    drawCaption(frame, tb3, tbCaption);

    if( nCurrentFrame > 180 ) {
        Rect tbGarbageBag = Rect(Point(440,110), Point(580,250));
        rectangle(frame, tbGarbageBag, RED, 2);
        drawCaption(frame, tbGarbageBag, "garbage bag");
    }

    int animStartFrame = 277;
    if( nCurrentFrame > animStartFrame
       && nCurrentFrame < 290) {
        Point tl = Point(340,80);
//        tl.y = (nCurrentFrame - animStartFrame) * 2;
        Point br = Point(460,140);
        Rect tbCardboardBox = Rect(tl, br);
        rectangle(frame, tbCardboardBox, BLACK, 2);
        drawCaption(frame, tbCardboardBox, "cardboard box");
    }

    
    cv::drawContours(frame, m_contours,
                     -1, // draw all contours
                     BLUE,
                     2); // with a thickness of 1

    vector<vector<Point> >::iterator itc = m_contours.begin();
    while( itc != m_contours.end() ) {
        vector<Point> _contour = *itc;

        RotatedRect rotatedRect = minAreaRect(Mat(_contour));
        if( verifySizes(rotatedRect) ) {
            Rect rect = boundingRect( _contour );
            rectangle(frame, rect.tl(), rect.br(), RED,
                      2, 8, 0 );
        }

        ++itc;
    }
    
}

/** @brief Draws a caption over passed rectangle: white on black
//
 */
void ClassicProcessor::drawCaption(Mat frame, Rect rect, string text) {

    rectangle(frame, Point(rect.x,rect.y - 10),
              Point(rect.x + rect.width, rect.y), BLACK, FILLED);
    putText(frame, text, Point(rect.x,rect.y),
            FONT_HERSHEY_SIMPLEX, 0.5, WHITE);
}

bool ClassicProcessor::verifySizes(RotatedRect rect) {
    
    float aspect = (float)rect.size.width / (float)rect.size.height;

    if( rect.size.height > m_minHeight
       && rect.size.width > m_minWidth )
//       && aspect < 1.0 )
        return true;

    return false;

}
