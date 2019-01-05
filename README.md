# Odette
This projects demonstrates object detection capabilities of SSD for constrained environemnts. On most checked devices, it is excepted to play about 30 fps. 

## How to build
OpenCV is the only dependency of this project. I build it with ver 3.4.3 but I belive the newest versions are goog also.
So inslall OpenCV in advance.
- on [RaspberryPi](https://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/)
- on macOS, just use <i>brew</i>
- on Windows, there is dedicated MSI.

This is XCode project, however, you'll be able to build it from code with the linked OpenCV libs:
- libopencv_core
- libopencv_imgproc
- libopencv_highgui (only for UI)
- libopencv_videoio
- libopencv_dnn

