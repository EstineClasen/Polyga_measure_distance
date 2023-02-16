#include "Measure.h"

// Constants.
const float f_confThreshold = 0.2f;
const float f_iouThreshold = 0.2f;

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

int main(int argc, char** argv)
{
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
	scanData currentScanData;
	memset(currentScanData.xyz_file, 0, sizeof(currentScanData.xyz_file));
	memset(currentScanData.depthImage_file, 0, sizeof(currentScanData.depthImage_file));
	memset(currentScanData.textureImage_file, 0, sizeof(currentScanData.textureImage_file));
	currentScanData = DepthImageData(*scanner);
	*/

	/////////Object detection on textureImage
	bool b_isGPU = false;
	const std::string sz_classNamesPath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\Polyga_measure_distance\\Models\\object_detection_classes.txt";
	const std::vector<std::string> vsz_classNamesList = detection_utils::loadClassNames(sz_classNamesPath);
	const std::string sz_imagePath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\Polyga_measure_distance\\Data\\1_not_padded.jpg";
	const std::string sz_modelPath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\Polyga_measure_distance\\Models\\turqoise5mm_tr528_val50_is736_b8_ep200_yolov5s.onnx";
	
	if (vsz_classNamesList.empty())
	{
		std::cerr << "Error: Empty class names file." << std::endl;
		return -1;
	}
	ObjectDetector od_detector{ nullptr };
	cv::Mat cv_m_imageInput;
	std::vector<Detection> vsd_result;

	try
	{
		od_detector = ObjectDetector(sz_modelPath, b_isGPU, cv::Size(736, 736));
		std::cout << "Model was initialized." << std::endl;

		cv_m_imageInput = cv::imread(sz_imagePath);
		vsd_result = od_detector.detect(cv_m_imageInput, f_confThreshold, f_iouThreshold);		//returns results with bbox scaled back to original image
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	detection_utils::visualizeDetection(cv_m_imageInput, vsd_result, vsz_classNamesList);

	cv::imshow("result", cv_m_imageInput);
	// cv::imwrite("result.jpg", image);
	cv::waitKey(0);

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

