//
//  HaarProcessor.hpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#ifndef HaarProcessor_hpp
#define HaarProcessor_hpp

#include "opencv2/objdetect/objdetect.hpp"

#include "IProcessor.hpp"

using namespace std;

class HaarProcessor : public IProcessor {
    
    private:
        CascadeClassifier m_boxes_cascade;
        vector<Rect>      m_boxes;
    
        void draw(Mat frame);
        
    
    public:
        HaarProcessor(string cascadeName, bool withTrace = true);
        void process(Mat frame, int nCurrentFrame,
                     bool withMultipleScale = false,
                     bool withDraw = true);
};

#endif /* HaarProcessor_hpp */
