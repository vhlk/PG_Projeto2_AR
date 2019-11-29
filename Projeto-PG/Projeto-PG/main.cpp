#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
using namespace cv;
using namespace std;

int main() {
	namedWindow("Teste", WINDOW_AUTOSIZE);
	Mat image1, image2, imageAux;
	//VideoCapture cap(0);
	VideoCapture cap = VideoCapture(0);
	if (!cap.isOpened()) { //verifica se cap abriu como esperado
		cout << "camera ou arquivo em falta";
		return 1;
	}
	image1 = imread("12.png", IMREAD_GRAYSCALE);

	if (image1.empty()) { //verifica a imagem1
		cout << "imagem 1 vazia";
		return 1;
	}

	int minValue = 50;
	int maxValue = 150;
	while (true) {
		cap >> image2;
		if (image2.empty()) {
			cout << "imagem 2 vazia";
			return 1;
		}

		//cvtColor(image2, image2, COLOR_BGR2GRAY);//coloca em grayscale

		vector<KeyPoint> kp1, kp2;
		Mat descriptor1, descriptor2;

		//Ptr<Feature2D> orb = xfeatures2d::SIFT::create(400);
		//Ptr<Feature2D> orb = xfeatures2d::SURF::create(400);
		Ptr<Feature2D> orb = ORB::create(400);
		orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);
		//Mat frame;
		//cap.read(frame);
		//imshow("frame", frame);

		namedWindow("out", CV_WINDOW_AUTOSIZE);
		createTrackbar("Min threshold: ", "out", &minValue, 500);
		createTrackbar("Max threshold: ", "out", &maxValue, 500);
		Mat blur;
		GaussianBlur(image2, blur, Size(5, 5), 0);
		Mat out;
		Canny(blur, out, minValue, maxValue);
		vector<vector<Point>> contorno;
		findContours(out, contorno, RETR_TREE, CHAIN_APPROX_NONE);
		drawContours(blur, contorno, -1, Scalar(0, 255, 0), 1);
		imshow("Blur", blur);

		imshow("out", out);
				
		drawKeypoints(image1, kp1, imageAux);
		//drawKeypoints(image2, kp2, image2);

		imshow("teste", image2);
		imshow("Teste", imageAux);
		if (waitKey(30) == 27) {
			break;
		}
	}
	cap.release();
	destroyAllWindows();
}
