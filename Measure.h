#pragma once

//Working with Polyga Scanner
#include <string.h>
#include <iostream>
#include <cstdio>
#include <stdio.h>      /* puts */
#include <time.h>       /* time_t, struct tm, time, localtime, strftime */
#include <fstream>
#include <format>
#include <windows.h>
#include "SBSDK.h"

//Working with OpenCV and ONNX Neural Network
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream> 
#include "include/detection_utils.h"
#include "include/detector.h"

// Namespaces.
using namespace std;
using namespace SBSDK3;
using namespace cv;
using namespace cv::dnn;


struct scanData
{
	char depthImage_file[100];
	char xyz_file[100];
	char textureImage_file[100];
};


class Measure
{
};

