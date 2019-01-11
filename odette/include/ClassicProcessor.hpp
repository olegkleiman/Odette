//
//  ClassicProcessor.hpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#ifndef ClassicProcessor_hpp
#define ClassicProcessor_hpp

#include "IProcessor.hpp"

using namespace std;


class ClassicProcessor : public IProcessor {
    
    private:
        vector<vector<Point>> m_contours;
        void draw(Mat frame, int nCurrentFrame);
        void drawCaption(Mat frame, Rect rect, string text);
        bool verifySizes(RotatedRect rect);
    
        int m_minHeight;
        int m_minWidth;
        bool m_withTrace;
        string m_winTraceName1;
        string m_winTraceName2;
    
    public:
        ClassicProcessor(bool withTrace = true);
        void process(Mat frame, int nCurrentFrame,
                     bool withMultipleScale = false,
                     bool withDraw = true);
};

#endif /* ClassicProcessor_hpp */
