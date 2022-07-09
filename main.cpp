#include <iostream>
#include <string>
#include <bits/stdc++.h>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\calib3d.hpp>

#include <ImgProcessFuncs.h>


int main() {
    // Create a matrix object and load the image to the matrix
    Mat original = imread("images/chicky_512.png", IMREAD_COLOR);

    // Check if image is not empty
    if (original.empty()){
        cout << "Couldn't load the image." << endl;
        system("pause");
        return-1;
    }

    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);

    // Rotate the image
    Mat rot = CVAlg::rotationMat(150.0);
    Mat trans = CVAlg::translateMat(500.0, 0.0);
    Mat combined = trans * rot;
    cout << "M = " << endl << " "  << combined << endl << endl;
    Mat result = CVAlg::backwardMapping(gray, combined, 2, 2);

    // Create a window with a specified name
    namedWindow("Target");

    // Display the image
    imshow("Target", result);
    waitKey(0);

    return 0;
}
