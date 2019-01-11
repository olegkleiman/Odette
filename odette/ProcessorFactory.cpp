//
//  ProcessorFactory.cpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#include "IProcessor.hpp"

#include "ProcessorFactory.hpp"
#include "HOGProcessor.hpp"
#include "ClassicProcessor.hpp"
#include "HaarProcessor.hpp"
#include "SSDProcessor.hpp"
#include "YOLOProcessor.hpp"

using namespace std;

IProcessor * ProcessorFactory::create(string method, bool withTrace) {
 
    if( method == "HOG" ) {
        return new _HOG(withTrace);
    } else if ( method == "Classic" ) {
        return new ClassicProcessor(withTrace);
    } else if( method == "Haar" ) {
        return new HaarProcessor("haarcascade_frontalface_default.xml", withTrace);
    } else if( method == "SSD" ) {
        return new SSDProcessor("./models/MobileNet/MobileNetSSD_deploy.prototxt.txt",
                                "./models/MobileNet/MobileNetSSD_deploy.caffemodel",
                                0.2 /*confidence*/);
    } else if( method == "YOLO" ) {
//        return new YOLOProcessor("./models/TensorFlow/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
//                                 "./models/TensorFlow/frozen_inference_graph.pb",
//                                 "./models/DarkNet/coco.names",
//                                 0.2);
        return new YOLOProcessor("./models/DarkNet/yolov3.cfg",
                                 "./models/DarkNet/yolov3.weights",
                                 "./models/DarkNet/coco.names",
                                 0.2 /*confidence*/);
    }
    
    return NULL;
}
