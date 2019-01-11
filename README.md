# Odette
This project demonstrates object detection capabilities of SSD/YOLOv3 for constrained environments. On most checked devices, it is excepted to play about 30 fps.

## How to build
OpenCV is the only dependency for this project. I built it with ver 4.0.1, but I believe the newest versions are good also.
Install OpenCV in advance:
- on [RaspberryPi](https://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/)
- on macOS, just use <i>brew</i>
- on Windows with VS, use <i>NuGet</i>

This is XCode project, however, you'll be able to build it from code with the linked OpenCV libs:
- libopencv_core
- libopencv_imgproc
- libopencv_imgcodecs
- libopencv_highgui (only for UI)
- libopencv_videoio
- libopencv_dnn
- libopencv_objdetect
- libopencv_dnn_objdetect

## MobileNets background
MobileNets differ from traditional CNNs through the usage of [depthwise separable convolution](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728).

The general idea behind depthwise separable convolution is to split convolution into two stages:
- A 3×3 depthwise convolution.
- Followed by a 1×1 pointwise convolution.
This allows to actually reduce the number of parameters in the network.

Very efficient and fast results are obtained when MobileNets is combined with SSD (Single Shot Detector) framework. In this project I use [Caffe implementation of SSD model](https://github.com/chuanqi305/MobileNet-SSD). One may use Caffe version of the [original TensorFlow implementation](https://github.com/Zehaos/MobileNet) 
