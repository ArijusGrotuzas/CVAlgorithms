#include <iostream>
#include <string>
#include <bits/stdc++.h>

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\calib3d.hpp>

#include <ImgProcessFuncs.h>

int main() {
    // Create a matrix object and load the image to the matrix
    Mat original = imread("images/blox.jpg", IMREAD_COLOR);

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
    Mat blurred = gausBlur(gray, 1, 1.5);

    // Get corner features of the image
    Mat response = shiTomasiCorners(blurred, 4, true);

    /*
    Mat corners;
    cv::threshold(result, corners, maximum * 0.9, 255, 0);
    */

    // Create a window with a specified name
    namedWindow("Target");

    // Display the image
    imshow("Target", response);
    waitKey(0);

    return 0;
}
