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
	Mat image1, image2, imageAux;
	VideoCapture cap(0);
	if (!cap.isOpened()) { //verifica se cap abriu como esperado
		cout << "camera ou arquivo em falta";
		return 1;
	}
	image1 = imread("imgPercyCort.jpg");
	
	if (image1.empty()) { //verifica a imagem1
		cout << "imagem 1 vazia";
		return 1;
	}
	//cout << "largura: " << image1.cols << "\n" << "altura : " << image1.rows;
	resize(image1, image1, Size(0,0), 0.1, 0.1);

	cout << "Defina o valor para Good Matches (valor entre 0.4 e 0.75 e preferivel), ou de enter para o padrao) \n";
	//definir threshold para o good matches
	string thresh;
	float ratio_thresh;
	if ('\n' == cin.get()) {
		ratio_thresh = 0.55f;
	}
	else {
		cin >> thresh;
		cout << thresh;
		ratio_thresh = stof(thresh);
		if (ratio_thresh > 1.0f || ratio_thresh < 0.0f) {
			cout << "Valor inválido! \n";
			return 1;
		}
	}
	namedWindow("Good Matches");
	int colorH = 255;
	int colorS = 255;
	int colorV = 0;
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
		Ptr<Feature2D> orb = xfeatures2d::SURF::create(400);
		//Ptr<Feature2D> orb = ORB::create(400);
		orb->detectAndCompute(image1, Mat(), kp1, descriptor1); //ira procurar pontos chaves
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);

		//usar o alg de Flann para comparar imagens
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		vector<vector<DMatch>> knn_matches;
		matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

		//usa um threshold -> diminui a taxa de erro e acerto
		vector<DMatch> good_matches;
		for (size_t i = 0; i < knn_matches.size(); i++)
		{
			if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		createTrackbar("H", "Good Matches", &colorH, 255);
		createTrackbar("S", "Good Matches", &colorS, 255);
		createTrackbar("V", "Good Matches", &colorV, 255);

		//Desenhar imagem de comparacao
		Mat img_matches;
		drawMatches(image1, kp1, image2, kp2, good_matches, img_matches, Scalar::all(-1),
			Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("Good Matches", img_matches);
		
		//verificar se houve matches
		if (good_matches.size() > 0) {
			vector<Point2f> ptsImg1;
			vector<Point2f> ptsImg2;
			for (size_t i = 0; i < good_matches.size(); i++)
			{
				//pegar os pontos a partir dos matches -> query e o 1 cara(descriptor1), train o 2 (descriptor2) do knnMatch
				ptsImg1.push_back(kp1[good_matches[i].queryIdx].pt);
				ptsImg2.push_back(kp2[good_matches[i].trainIdx].pt);
			}

			Mat h = findHomography(ptsImg1, ptsImg2, RANSAC);

			//pegar vértices da imagem
			vector<Point2f> verticesImg(4);
			verticesImg[0] = Point2f(0, 0);
			verticesImg[1] = Point2f((float)image1.cols, 0);
			verticesImg[2] = Point2f((float)image1.cols, (float)image1.rows);
			verticesImg[3] = Point2f(0, (float)image1.rows);

			if (!h.empty()) {
				vector<Point2f> verticesCam(4);
				perspectiveTransform(verticesImg, verticesCam, h);
				line(img_matches, verticesCam[0] + Point2f(image1.cols, 0), verticesCam[1] + Point2f(image1.cols, 0), Scalar((double)colorH, (double)colorS, (double)colorV), 5);
				line(img_matches, verticesCam[1] + Point2f(image1.cols, 0), verticesCam[2] + Point2f(image1.cols, 0), Scalar((double)colorH, (double)colorS, (double)colorV), 5);
				line(img_matches, verticesCam[2] + Point2f(image1.cols, 0), verticesCam[3] + Point2f(image1.cols, 0), Scalar((double)colorH, (double)colorS, (double)colorV), 5);
				line(img_matches, verticesCam[3] + Point2f(image1.cols, 0), verticesCam[0] + Point2f(image1.cols, 0), Scalar((double)colorH, (double)colorS, (double)colorV), 5);
			}
			
			imshow("Good Matches", img_matches);
		}



		/* mostrar taxa de acerto
		double qtdPontos;
		if (kp1.size() < kp2.size()) {
			qtdPontos = kp1.size();
		}
		else {
			qtdPontos = kp2.size();
		}
		cout << "Taxa de similaridade: " << (good_matches.size() / qtdPontos) << "\n";
		*/

		//drawKeypoints(image1, kp1, imageAux);
		//drawKeypoints(image2, kp2, image2);
		//imshow("teste", image2);
		//imshow("Teste", image1);
		if (waitKey(30) == 27) {
			break;
		}
	}
	destroyAllWindows();
}
