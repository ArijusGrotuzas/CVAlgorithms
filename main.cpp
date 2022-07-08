#include <iostream>
#include <string>
#include <bits/stdc++.h>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\calib3d.hpp>

#include <ImgProcessFuncs.h>

int main() {
    // Create a matrix object and load the image to the matrix
    Mat original = imread("images/fruits.jpg", IMREAD_COLOR);

    // Check if image is not empty
    if (original.empty()){
        cout << "Couldn't load the image." << endl;
        system("pause");
        return-1;
    }

    // Convert the image to grayscale
    Mat gray = grayscale(original);

    // Stretch the image's histogram
    histogramStretch(gray);

    // Blur the image
    Mat rot = shearMat(0.0, 1.1);
    Mat rotated = backwardMapping(gray, rot, 1, 2.5);

    Mat result;
    cv::resize(rotated, result, cv::Size(), 0.5, 0.5);

    // Create a window with a specified name
    namedWindow("Target");

    // Display the image
    imshow("Target", result);
    waitKey(0);

    return 0;
}
