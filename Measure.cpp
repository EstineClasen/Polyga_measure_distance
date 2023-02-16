#include "Measure.h"

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);

std::string ScannerModelString(ISBScanner& scanner) {
	int model = (int)scanner.getScannerModel();
	switch (model) {
	case (int)SBScannerModelEnums::POLYGA_SCANNER:
		return "POLYGA_SCANNER";
	case (int)SBScannerModelEnums::L6:
		return "L6";
	case (int)SBScannerModelEnums::S1:
		return "S1";
	case (int)SBScannerModelEnums::S5:
		return "S5";
	case (int)SBScannerModelEnums::H3:
	case (int)SBScannerModelEnums::C210:
		return "C210";
	case (int)SBScannerModelEnums::C506:
		return "C506";
	case (int)SBScannerModelEnums::C504:
		return "C504";
	default:
		return "POLYGA_SCANNER";

	}
}
void PrintDeviceInfo(ISBScanner& scanner)
{

	int serial = 0;
	serial = scanner.getSerial();
	std::string model = ScannerModelString(scanner);
	std::cout << "======Device Information======" << endl;
	std::cout << "Scanner Model: " << model << endl;
	std::cout << "Serial Number: " << serial << endl;
	std::cout << "==============================" << endl;

	return;
}
void PrintMeshInfo(SBMesh mesh)
{
	int nVertices = mesh.getNumVertices();
	SBVector* vertices = new SBVector[nVertices];
	mesh.getVertices(vertices);


	int nFaces = mesh.getNumFaces();
	SBFace* faces = new SBFace[nFaces];
	mesh.getFaces(faces);

	std::cout << "========== Mesh Info ==========" << endl;
	std::cout << "num Vertices: " << nVertices << endl;
	std::cout << "num Faces: " << nFaces << endl;
	std::cout << "===============================" << endl;

	delete[] vertices;
	delete[] faces;
}

void ScanExample(ISBScanner& scanner)
{

	//create handle to get mesh data.
	SBMesh mesh;
	SBScan scan;
	std::cout << "Start Scan" << std::endl;

	if (scanner.scan(SBScanParams(), scan, mesh) != SBStatus::OK)
	{
		std::cout << "Scan Failed" << std::endl;
		return;
	}

	PrintMeshInfo(mesh);

	//Pathes to save data
	//std::string scanFolder = "Scan";
	std::string scanFolder = "C:\\Users\\estin\\OneDrive\\Documents\\Estine\\Ortho-Design\\Code\\C++\\Polyga\\Polyga_measure_distance\\Exports";

	//Save mesh in ply
	std::string meshFileName(scanFolder + "\\mesh.ply");

	std::cout << "Saving mesh to " << scanFolder << std::endl;
	mesh.save(meshFileName.c_str());
	return;
}


void PrintVImageInfo(SBImage& vImage) {
	int width = vImage.width();
	int height = vImage.height();
	int bytesPerPixel = vImage.bytePerPixel();



	std::cout << "========== Depth Image Info ==========" << endl;
	std::cout << "Width: " << width << endl;
	std::cout << "Height: " << height << endl;
	std::cout << "Bytes per pixel: " << bytesPerPixel << endl;
	std::cout << "======================================" << endl;

}

void SaveTextureImage(ISBScanner& scanner)
{
	// Capturing Scan Images
	SBScan scan;
	std::cout << "Start Scan" << std::endl;

	SBMesh mesh;
	if (scanner.scan(SBScanParams(), scan, mesh) != SBStatus::OK)
	{
		std::cout << "Scan Failed in SaveTextureImage" << std::endl;
		return;
	}

	// Get DateTime to save files correctly
	time_t t_currentRawTime;
	struct tm tm_timeinfo;
	char ca_datetime[50];
	time(&t_currentRawTime);
	//timeinfo = localtime(&t_currentRawTime);
	localtime_s(&tm_timeinfo, &t_currentRawTime);
	strftime(ca_datetime, sizeof(ca_datetime), "%Y%m%d\_%Hh%Mm%S", &tm_timeinfo);
	std::cout << ca_datetime << std::endl;

	char ca_textureImageFile[100];
	memset(ca_textureImageFile, 0, sizeof(ca_textureImageFile));
	int retVal1 = snprintf(ca_textureImageFile, sizeof(ca_textureImageFile), "Exports/TextureImage_%s.jpg", ca_datetime);
	if (retVal1 > 0 && retVal1 < sizeof(ca_textureImageFile))
	{
		cout << ca_textureImageFile << " write success" << endl;
	}
	else
		cout << "Error writing to ca_textureImageFile" << endl;

	//Only get training images
	SBImage textureImage;
	scan.getTextureImage(textureImage);
	textureImage.save(ca_textureImageFile);

	return;
}

scanData DepthImageData(ISBScanner& scanner)
{
	// Capturing Scan Images
	SBScan scan;
	std::cout << "Start Scan" << std::endl;

	scanData newScanData;
	memset(newScanData.xyz_file, 0, sizeof(newScanData.xyz_file));
	memset(newScanData.depthImage_file, 0, sizeof(newScanData.depthImage_file));
	memset(newScanData.textureImage_file, 0, sizeof(newScanData.textureImage_file));
	
	SBMesh mesh;
	if (scanner.scan(SBScanParams(), scan, mesh) != SBStatus::OK)
	{
		std::cout << "Scan Failed" << std::endl;
		return newScanData;
	}

	// Get DateTime to save files correctly
	time_t t_currentRawTime;
	struct tm tm_timeinfo;
	char ca_datetime[50];
	time(&t_currentRawTime);
	//timeinfo = localtime(&t_currentRawTime);
	localtime_s(&tm_timeinfo, &t_currentRawTime);
	strftime(ca_datetime, sizeof(ca_datetime), "%Y%m%d\_%Hh%Mm%S", &tm_timeinfo);
	std::cout << ca_datetime << std::endl;
	
	char ca_xyzFile[100];
	memset(ca_xyzFile, 0, sizeof(ca_xyzFile));
	char ca_depthImageFile[100];
	memset(ca_depthImageFile, 0, sizeof(ca_depthImageFile));
	char ca_textureImageFile[100];
	memset(ca_textureImageFile, 0, sizeof(ca_textureImageFile));
	int retVal1 = snprintf(ca_xyzFile, sizeof(ca_xyzFile), "Exports/xyz_%s.xyz", ca_datetime);
	if (retVal1 > 0 && retVal1 < sizeof(ca_xyzFile))
	{
		cout << ca_xyzFile << " write success" << endl;
	}
	else
		cout << "Error writing to ca_xyzFile" << endl;
	int retVal2 = snprintf(ca_depthImageFile, sizeof(ca_depthImageFile), "Exports/DepthImage_%s.jpg", ca_datetime);
	if (retVal2 > 0 && retVal2 < sizeof(ca_depthImageFile))
	{
		cout << ca_depthImageFile << " write success" << endl;
	}
	else
		cout << "Error writing to ca_depthImageFile" << endl;
	int retVal3 = snprintf(ca_textureImageFile, sizeof(ca_textureImageFile), "Exports/TextureImage_%s.jpg", ca_datetime);
	if (retVal3 > 0 && retVal3 < sizeof(ca_textureImageFile))
	{
		cout << ca_textureImageFile << " write success" << endl;
	}
	else
		cout << "Error writing to ca_textureImageFile" << endl;
	
	ofstream of;
	//of.open("out_20230206.xyz");
	of.open(ca_xyzFile);
	SBImage depthImage = mesh.mVImage;
	PrintVImageInfo(depthImage);
	uint8_t* depthData = depthImage.data();
	for (int j = 0; j < depthImage.height(); j++) {
		int offset = j * depthImage.width() * depthImage.bytePerPixel();
		for (int i = 0; i < depthImage.width(); i++) {

			int iter = offset + (depthImage.bytePerPixel() * i);

			//float x = *((float*)&depthData[ iter]);
			//float y = *((float*)&depthData[iter + 4]);
			//float z = *((float*)&depthData[iter + 8]);
			float x = *((float*)(depthData + iter));
			float y = *((float*)(depthData + iter + 4));
			float z = *((float*)(depthData + iter + 8));
			if (!(isnan(z) || isnan(y) || isnan(x)))
			{
				//of << x << " " << y << " " << z << "\n";;				//EC_20230202
				//written += 1;			//EC_20230202
			}
			else                 //EC_20230202
			{
				x = -100;
				y = -100;
				z = -100;
			}
			of << x << " " << y << " " << z << "\n";;
		}
	}
	of.close();

	depthImage.save(ca_depthImageFile);
	SBImage textureImage;
	scan.getTextureImage(textureImage);
	textureImage.save(ca_textureImageFile);
	memcpy(newScanData.xyz_file, ca_xyzFile, sizeof(ca_xyzFile));
	memcpy(newScanData.depthImage_file, ca_depthImageFile, sizeof(ca_depthImageFile));
	memcpy(newScanData.textureImage_file, ca_textureImageFile, sizeof(ca_textureImageFile));

	return newScanData;
}

void draw_label(Mat& input_image, string label, int left, int top)
{
	// Display the label at the top of the bounding box.
	int baseLine;
	Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
	top = max(top, label_size.height);
	// Top left corner.
	Point tlc = Point(left, top);
	// Bottom right corner.
	Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
	// Draw white rectangle.
	rectangle(input_image, tlc, brc, BLACK, FILLED);
	// Put the label on the black rectangle.
	putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(Mat& input_image, Net& net)
{
	// Convert to blob.
	Mat blob;
	blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

	net.setInput(blob);

	// Forward propagate.
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	return outputs;
}

Mat post_process(Mat& input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
	// Initialize vectors to hold respective outputs while unwrapping     detections.
	vector<int> class_ids;
	vector<float> confidences;
	vector<Rect> boxes;
	// Resizing factor.
	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;
	float* data = (float*)outputs[0].data;
	const int dimensions = 85;
	// 25200 for default size 640.
	const int rows = 25200;
	// Iterate through 25200 detections.
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];
		// Discard bad detections and continue.
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float* classes_scores = data + 5;
			// Create a 1x85 Mat and store class scores of 80 classes.
			Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
			// Perform minMaxLoc and acquire the index of best class  score.
			Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			// Continue if the class score is above the threshold.
			if (max_class_score > SCORE_THRESHOLD)
			{
				// Store class ID and confidence in the pre-defined respective vectors.
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
				// Center.
				float cx = data[0];
				float cy = data[1];
				// Box dimension.
				float w = data[2];
				float h = data[3];
				// Bounding box coordinates.
				int left = int((cx - 0.5 * w) * x_factor);
				int top = int((cy - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				// Store good detections in the boxes vector.
				boxes.push_back(Rect(left, top, width, height));
			}
		}
		// Jump to the next row.
		data += 85;
	}
	// Perform Non-Maximum Suppression and draw predictions.
	vector<int> indices;
	NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		int left = box.x;
		int top = box.y;
		int width = box.width;
		int height = box.height;
		// Draw bounding box.
		rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);
		// Get the label for the class name and its confidence.
		string label = format("%.2f", confidences[idx]);
		label = class_name[class_ids[idx]] + ":" + label;
		// Draw class labels.
		draw_label(input_image, label, left, top);
	}
	return input_image;
}

void createGraphTextFromTFModel()
{
	writeTextGraph("Models/YOLOv5s.pb", "Models/yolov5_20230214.pbtxt");
	return;
}

vector<string> load_class_list()
{
	//string file_path = "C:/Users/estin/PycharmProjects/TrainYolov5/data/5mm_turqoise/";
	vector<string> class_list;
	//ifstream ifs(string(file_path + "object_detection_classes.txt").c_str());
	ifstream ifs(string("object_detection_classes_coco.txt").c_str());
	string line;
	while (getline(ifs, line))
	{
		class_list.push_back(line);
	}
	return class_list;
}

void load_net(Net& net, bool is_cuda)
{
	//https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c	Can make this faster
	string fileName = "Models/YOLOv5s.onnx";
	struct stat buffer;
	cout << (stat(fileName.c_str(), &buffer) == 0);
	
	//auto result = readNet("C:\\Users\\estin\\PycharmProjects\\TrainYolov5\\YOLOv5\\models\\YOLOv5s.onnx");
	auto result = readNetFromONNX(fileName);
	//auto result = readNet("Models\\YOLOv5s.pb", "Models\\yolov5_20230214.pbtxt", "TensorFlow");
	if (is_cuda)
	{
		cout << "Attempting to use CUDA\n";
		result.setPreferableBackend(DNN_BACKEND_CUDA);
		result.setPreferableTarget(DNN_TARGET_CUDA_FP16);
	}
	else
	{
		cout << "Running on CPU\n";
		result.setPreferableBackend(DNN_BACKEND_OPENCV);
		result.setPreferableTarget(DNN_TARGET_CPU);
	}
	net = result;
}

int main(int argc, char** argv)
{
	//createGraphTextFromTFModel();
	/*
	//Initialization
	ISBScanner* scanner;
	SBDeviceList devices;

	//Checking for connected devices
	SBFactory::getDevices(devices);
	if (devices.size() == 0)
	{
		std::cout << "No Device Detected" << std::endl;
		return 0;
	}

	//Creating scanner instance using the first device detected    
	scanner = SBFactory::createDevice(devices[0]);
	//Stablish connection with scanner
	if (scanner->connect() != SBStatus::OK)
	{
		std::cout << "Failed to connect " << std::endl;
		cout << "Press enter to exit";
		getchar();
		return 0;
	}

	cout << "Scanner Connected" << std::endl;

	PrintDeviceInfo(*scanner);

	//Simple Scan
	//ScanExample(*scanner);

	//Only get texture image to save for training NN
	//SaveTextureImage(*scanner);

	//Depth Image to xyz and save textureImage
	//scanData currentScanData;
	//memset(currentScanData.xyz_file, 0, sizeof(currentScanData.xyz_file));
	//memset(currentScanData.depthImage_file, 0, sizeof(currentScanData.depthImage_file));
	//memset(currentScanData.textureImage_file, 0, sizeof(currentScanData.textureImage_file));
	//currentScanData = DepthImageData(*scanner);
	*/

	/////////Object detection on textureImage
	// https://www.youtube.com/watch?v=tjzJs9rOeEM and https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/ and https://github.com/niconielsen32/ComputerVision/blob/master/OpenCVdnn/objectDetection.cpp
	// Load class list.
	vector<string> class_list = load_class_list();
	// Load image.
	Mat frame;
	frame = imread("Data/traffic.jpeg");
	int image_height = frame.cols;
	int image_width = frame.rows;
	// Load model.
	bool is_cuda = false;		//argc > 1 && strcmp(argv[1], "cuda") == 0;
	Net net;
	load_net(net, is_cuda);
	//std::filesystem::path _model = "C:/Users/estin/PycharmProjects/TrainYolov5/YOLOv5/models/YOLOv5s.onnx";
	//net = readNet("C:/Users/estin/PycharmProjects/TrainYolov5/YOLOv5/models/YOLOv5s.onnx");
	//net = readNetFromONNX("C:\\Users\\estin\\PycharmProjects\\TrainYolov5\\YOLOv5\\models\\YOLOv5s.onnx");
	// Set a min confidence score for the detections
	float min_confidence_score = 0.5;
	auto start = getTickCount();
	// Create a blob from the image
	Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),true, false);
	// Set the blob to be input to the neural network
	net.setInput(blob);
	// Forward pass of the blob through the neural network to get the predictions
	Mat output = net.forward();
	auto end = getTickCount();

	// Matrix with all the detections
	Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());

	// Run through all the predictions
	for (int i = 0; i < results.rows; i++) {
		int class_id = int(results.at<float>(i, 1));
		float confidence = results.at<float>(i, 2);

		// Check if the detection is over the min threshold and then draw bbox
		if (confidence > min_confidence_score) {
			int bboxX = int(results.at<float>(i, 3) * frame.cols);
			int bboxY = int(results.at<float>(i, 4) * frame.rows);
			int bboxWidth = int(results.at<float>(i, 5) * frame.cols - bboxX);
			int bboxHeight = int(results.at<float>(i, 6) * frame.rows - bboxY);
			rectangle(frame, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0, 0, 255), 2);
			string class_name = class_list[class_id - 1];
			putText(frame, class_name + " " + to_string(int(confidence * 100)) + "%", Point(bboxX, bboxY - 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
		}
	}

	auto totalTime = (end - start) / getTickFrequency();


	putText(frame, "FPS: " + to_string(int(1 / totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);

	imshow("image", frame);


	waitKey(0);

	//Find center of bounding box VS find max Z in that region
	

	//Get coordinates for each pin center from .xyz row
	//Row = (x+1)+(720*y) (check if array indexed at 0 or 1 like file)

	//Calculate Euclidean distance between points

	//Disconnecting scanner
	//scanner->disconnect();
	//Disposing scanner pointer since SBFactory::createDevice() news the scanner pointer.
	//SBFactory::disposeDevice(scanner);
	cout << "Press enter to exit";
	getchar();

	return 0;
}

