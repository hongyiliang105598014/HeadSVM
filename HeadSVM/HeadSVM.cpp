// HeadSVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\core.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>

#define FEATURESAMOUNT 324
#define POSITIVESAMPLEAMOUNT 1383
#define NEGATIVESAMPLEAMOUNT 1667
#define WINDOW_SIZE cv::Size(32, 32)

using namespace cv;
using namespace cv::ml;

int main()
{
	char filepath[100];
	//cv::Mat image = cv::Mat(WINDOW_SIZE, CV_8UC1);
	cv::Mat	image;
	cv::Mat trainData = cv::Mat(cv::Size(FEATURESAMOUNT, POSITIVESAMPLEAMOUNT + NEGATIVESAMPLEAMOUNT), CV_32FC1);
	cv::Mat responses = cv::Mat(cv::Size(1, POSITIVESAMPLEAMOUNT + NEGATIVESAMPLEAMOUNT), CV_32FC1);
	cv::HOGDescriptor hogDescriptor = cv::HOGDescriptor(WINDOW_SIZE, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	std::vector<float> features;
	//cv::SVMParams params;
	//params.kernel_type = cv::SVM::LINEAR;
	//params.svm_type = cv::SVM::C_SVC;
	//params.C = 1;
	//params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	//cv::SVM svm;

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setC(1);

	cv::Mat forTesting = cv::Mat(cv::Size(FEATURESAMOUNT, 1), CV_32FC1);
	//cv::Mat imageShow = cv::Mat(cv::Size(32, 32), CV_8UC3);
	cv::Mat imageShow;
	int errorCount = 0;

	float tempResponse = 0;
	bool firstGet = true;
	float maxResponse = 0;
	cv::Rect maxResponsePosition;
	float maxResponseScale = 0;
	cv::Size sourceSize;

	////正樣本
	//for (int index = 1; index <= POSITIVESAMPLEAMOUNT; index++)
	//{
	//	features.clear();
	//	sprintf(filepath, "dataset_motor_front_end\\TrainingSet\\end\\end (%d).jpg", index);
	//	image = cv::imread(filepath, 0);
	//	hogDescriptor.compute(image, features, cv::Size(0, 0), cv::Size(0, 0));
	//	for (int descriptorIndex = 0; descriptorIndex < FEATURESAMOUNT; descriptorIndex++)
	//		trainData.ptr<float>(index - 1)[descriptorIndex] = features[descriptorIndex];
	//	responses.ptr<float>(index - 1)[0] = 1.0;
	//}
	////負樣本
	//for (int index = 1; index <= NEGATIVESAMPLEAMOUNT; index++)
	//{
	//	features.clear();
	//	sprintf(filepath, "dataset_motor_front_end\\neg_front_back\\neg (%d).jpg", index);
	//	image = cv::imread(filepath, 0);
	//	hogDescriptor.compute(image, features, cv::Size(0, 0), cv::Size(0, 0));
	//	for (int descriptorIndex = 0; descriptorIndex < FEATURESAMOUNT; descriptorIndex++)
	//		trainData.ptr<float>(POSITIVESAMPLEAMOUNT + index - 1)[descriptorIndex] = features[descriptorIndex];
	//	responses.ptr<float>(POSITIVESAMPLEAMOUNT + index - 1)[0] = -1.0;
	//}
	//svm.train_auto(trainData, responses, cv::Mat(), cv::Mat(), params);
	//svm.save("dataset_motor_front_end\\motor_back.xml");
	////訓練樣本測試
	//std::cout << "Positive testing" << std::endl;
	//for (int index = 1; index < POSITIVESAMPLEAMOUNT; index++)
	//{
	//	features.clear();
	//	sprintf(filepath, "dataset_motor_front_end\\TrainingSet\\end\\end (%d).jpg", index);
	//	image = cv::imread(filepath, 0);
	//	hogDescriptor.compute(image, features, cv::Size(0, 0), cv::Size(0, 0));
	//	for (int descriptorIndex = 0; descriptorIndex < FEATURESAMOUNT; descriptorIndex++)
	//		forTesting.ptr<float>(0)[descriptorIndex] = features[descriptorIndex];
	//	//std::cout << index << " " << svm.predict(forTesting, false) << std::endl;
	//	if (svm.predict(forTesting, false) != 1)
	//		errorCount++;
	//}
	//std::cout << "POS Error Count : " << errorCount << std::endl;
	//errorCount = 0;
	//std::cout << "Negative testing" << std::endl;
	//for (int index = 1; index < NEGATIVESAMPLEAMOUNT; index++)
	//{
	//	features.clear();
	//	sprintf(filepath, "dataset_motor_front_end\\neg_front_back\\neg (%d).jpg", index);
	//	image = cv::imread(filepath, 0);
	//	hogDescriptor.compute(image, features, cv::Size(0, 0), cv::Size(0, 0));
	//	for (int descriptorIndex = 0; descriptorIndex < FEATURESAMOUNT; descriptorIndex++)
	//		forTesting.ptr<float>(0)[descriptorIndex] = features[descriptorIndex];
	//	//std::cout << index << " " << svm.predict(forTesting, false) << std::endl;
	//	if (svm.predict(forTesting, false) != -1)
	//		errorCount++;
	//}
	//std::cout << "NEG Error Count : " << errorCount << std::endl;
	//system("pause");
	//實際測試


	//image = cv::imread("head/image (901).jpg");
	//cv::imshow("test", image);
	//cv::waitKey(0);


	svm->load("head.xml");
	errorCount = 0;
	for (int imageIndex = 902; imageIndex <= 1533; imageIndex++)
	{
		
		printf("%d\n", imageIndex);
		sprintf(filepath, "head\\image (%d).jpg", imageIndex);
		//imageShow.release();
		//imageShow.create(WINDOW_SIZE, CV_8UC3);
		//image.release();
		//image.create(WINDOW_SIZE, CV_8UC1);
		image = cv::imread(filepath, 0);
		sourceSize = image.size();
		for (float scale = 0.6; scale <= 1.8; scale = scale + 0.4)
		{
			image = cv::imread(filepath, 0);
			
			cv::resize(image, image, cv::Size(0, 0), scale, scale);

			//cv::imshow("test", image);
			//std::cout << image.rows << " " << image.cols << std::endl;
			//cv::waitKey(0);


			for (int y = 0; y + 31 < image.rows; y++)
			{
				for (int x = 0; x + 31 < image.cols; x++)
				{
					cv::Mat ROI = cv::Mat(image, cv::Rect(x, y, 32, 32));

					cv::imshow("ROI", ROI);
					cv::waitKey(0);
					//std::vector<cv::Point> positions;
					//positions.push_back(cv::Point(image.cols / 2 , image.rows / 2));
					
					hogDescriptor.compute(ROI, features, cv::Size(0, 0), cv::Size(0, 0));
					cv::waitKey(0);
					for (int descriptorIndex = 0; descriptorIndex < FEATURESAMOUNT; descriptorIndex++)
						forTesting.ptr<float>(0)[descriptorIndex] = features[descriptorIndex];
					tempResponse = svm->predict(forTesting);
					if (tempResponse < 0)
					{
						if (firstGet == true || abs(tempResponse) > maxResponse)
						{
							maxResponseScale = scale;
							maxResponse = abs(tempResponse);
							maxResponsePosition = cv::Rect(x, y, 32, 32);
						}
						firstGet = false;
					}
					features.clear();
				}
			}
		}
		if (firstGet != true)
		{
			imageShow = cv::imread(filepath, 1);
			cv::resize(imageShow, imageShow, cv::Size(0, 0), maxResponseScale, maxResponseScale);
			cv::rectangle(imageShow, maxResponsePosition, cv::Scalar(0, 0, 255));
			cv::resize(imageShow, imageShow, sourceSize);
		}
		sprintf(filepath, "head_result\\image (%d).jpg", imageIndex);
		cv::imwrite(filepath, imageShow);
		firstGet = true;
	}
	system("pause");
	return 0;
}

