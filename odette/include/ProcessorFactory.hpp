//
//  ProcessorFactory.hpp
//  odette
//
//  Created by Oleg Kleiman on 08/01/2019.
//  Copyright © 2019 Oleg Kleiman. All rights reserved.
//

#ifndef ProcessorFactory_hpp
#define ProcessorFactory_hpp

#include <iostream>
#include <opencv2/core.hpp>
#include "IProcessor.hpp"

using namespace std;

class ProcessorFactory {
    public:
        static IProcessor* create(string method, bool withTrace = true);
};

#endif /* ProcessorFactory_hpp */
