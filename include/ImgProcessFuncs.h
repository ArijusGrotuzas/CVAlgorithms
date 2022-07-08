#ifndef IMGPROCESSFUNCS_H
#define IMGPROCESSFUNCS_H

#include <math.h>

#define pi 3.14159

using namespace std;
using namespace cv;

namespace CVAlg{

    /**< Mathematical functions */

    float euclideanDist(Point a, Point b){
        float dist = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
        return dist;
    }

    float manhattanDist(Point a, Point b){
        float dist = abs(a.x - b.x) + abs(a.x - b.y);
        return dist;
    }

    float lerp(float a, float b, float t){
        return a + t * (b - a);
    }

    double median(vector<int>& arr, int len){
        sort(arr.begin(), arr.end());

        if(len % 2 != 0)
            return (double)arr[len / 2];

        return (double)(arr[(len - 1) / 2] + arr[len / 2]) / 2.0;
    }

    /**< Matrices for performing 2D transformations of images*/

    Mat scaleMat(float scaleX, float scaleY){

        Mat trans(3, 3, CV_32FC1, Scalar(0));

        trans.at<float>(0, 0) = scaleX;
        trans.at<float>(1, 1) = scaleY;
        trans.at<float>(2, 2) = 1.0;

        return trans;
    }

    Mat rotationMat(float theta){

        Mat trans(3, 3, CV_32FC1, Scalar(0));

        trans.at<float>(0, 0) = cos(theta);
        trans.at<float>(0, 1) = -sin(theta);
        trans.at<float>(1, 0) = sin(theta);
        trans.at<float>(1, 1) = cos(theta);
        trans.at<float>(2, 2) = 1.0;

        return trans;
    }

    Mat shearMat(float sX, float sY){

        Mat trans(3, 3, CV_32FC1, Scalar(0));

        trans.at<float>(0, 0) = 1.0;
        trans.at<float>(0, 1) = sX;
        trans.at<float>(1, 0) = sY;
        trans.at<float>(1, 1) = 1.0;
        trans.at<float>(2, 2) = 1.0;

        return trans;

    }

    /**< Kernels for convolutions */

    Mat gaussianKernel(int radius, float sigma){
        float r, sum = 0.0, s = 2 * sigma * sigma;

        // Initialize empty single-channel matrix
        Mat result(2 * radius + 1, 2 * radius + 1, CV_32FC1, Scalar(0));

        // Draw values from 2D gaussian distribution
        for (int x = -radius; x <= radius; x++){
            for (int y = -radius; y <= radius; y++){

                // Calculate gaussian value and assign it to resulting matrix
                r = sqrt(x * x + y * y);
                float gaus = exp(-r / s) / (pi * s);
                result.at<float>(x + radius, y + radius) = gaus;
                sum += gaus;
            }
        }

        // Normalize the kernel
        for (int i = 0; i < result.rows; i++){
            for (int j = 0; j < result.cols; j++){
                result.at<float>(i, j) /= sum;
            }
        }

        // Return gaussian kernel
        return result;
    }

    Mat edgeHorizontal(){
        Mat trans(1, 3, CV_32FC1, Scalar(0));

        trans.at<float>(0, 0) = -1;
        trans.at<float>(0, 2) = 1;

        return trans;
    }

    Mat edgeVertical(){
        Mat trans(3, 1, CV_32FC1, Scalar(0));

        trans.at<float>(0, 0) = -1;
        trans.at<float>(2, 0) = 1;

        return trans;
    }

    Mat sobelVertical(){
        Mat sobelK(3, 3, CV_32FC1, Scalar(0));

        sobelK.at<float>(0, 0) = -1.0;
        sobelK.at<float>(0, 1) = -2.0;
        sobelK.at<float>(0, 2) = -1.0;

        sobelK.at<float>(2, 0) = 1.0;
        sobelK.at<float>(2, 1) = 2.0;
        sobelK.at<float>(2, 2) = 1.0;

        return sobelK;
    }

    Mat sobelHorizontal(){
        Mat sobelK(3, 3, CV_32FC1, Scalar(0));

        sobelK.at<float>(0, 0) = -1.0;
        sobelK.at<float>(1, 0) = -2.0;
        sobelK.at<float>(2, 0) = -1.0;

        sobelK.at<float>(0, 2) = 1.0;
        sobelK.at<float>(1, 2) = 2.0;
        sobelK.at<float>(2, 2) = 1.0;

        return sobelK;
    }

    /**< Point processing operations */

    Mat grayscale(Mat& img){

        // Get number of channels of the input image
        int channels = img.channels();

        // Check if input image is a colored image
        if(channels != 3){
            cout << "Error, expected a colored image, returning empty image..." << endl;
            return Mat(0, 0, CV_8UC1, Scalar(0));
        }

        int sum, rows = img.rows, cols = img.cols;

        // Initialize empty single-channel result image
        Mat result(rows, cols, CV_8UC1, Scalar(0));

        // Loop through all the pixels in the image
        for(int x = 0; x < rows; x++){
            for(int y = 0; y < cols; y++){
                sum = 0;

                // Loop through all the channel values
                for(int i = 0; i < channels; i ++){
                    unsigned char *pixelPtr = img.ptr(x) + (y * channels) + i;
                    int val = *pixelPtr;
                    sum += val;
                }

                // Set the output value to be an average of all input image's channel values
                result.data[result.step[0]*x + result.step[1]* y] = sum / channels;
            }
        }

        return result;
    }

    Mat threshold(Mat& img, int limit){
        // Get number of channels of the input image
        int channels = img.channels();

        // Check if input image is a colored image
        if(channels != 1){
            cout << "Error, expected a grayscale image, returning empty image..." << endl;
            return Mat(0, 0, CV_8UC1, Scalar(0));
        }

        int rows = img.rows, cols = img.cols;

        // Initialize empty single-channel result image
        Mat result(rows, cols, CV_8UC1, Scalar(0));

        // Loop through all the pixels in the image
        for(int x = 0; x < rows; x++){
            for(int y = 0; y < cols; y++){

                unsigned char *pixelPtr = img.ptr(x) + y;
                int val = *pixelPtr;

                if(val > limit){
                    result.data[result.step[0]*x + result.step[1]* y] = 255;
                }
                else{
                    result.data[result.step[0]*x + result.step[1]* y] = 0;
                }
            }
        }

        return result;
    }

    /**< Neighborhood processing operations*/

    Mat medianFilter(Mat& img, int r){

        // Get number of channels of the input image
        int channels = img.channels();

        // Check if input image is a colored image
        if(channels != 1){
            cout << "Error, expected a grayscale image..." << endl;
        }

        // Get image resolution
        int rows = img.rows, cols = img.cols;

        // Initialize empty single-channel result image
        Mat result(rows, cols, CV_8UC1, Scalar(0));

        // Initialize a list of neighbors
        vector<int> neighbours;

        // Loop through all the pixels in the image
        for(int x = 0; x < rows; x++){
            for(int y = 0; y < cols; y++){

                // Loop through the neighboring pixels
                for(int i = x - r; i <= x + r; i++){
                    for(int j = y - r; j <= y + r; j++){

                        // Check if outside image bounds
                        if(i < 0 || i > rows || j < 0 || j > cols){
                            // Skip iteration
                            continue;
                        }
                        else{
                            // Get pixel value
                            unsigned char *pixelPtr = img.ptr(i) + j;
                            neighbours.push_back((int)*pixelPtr);
                        }
                    }
                }

                // Get the median value of all pixel values
                double res = median(neighbours, neighbours.size());
                result.data[result.step[0]*x + result.step[1]* y] = (int)round(res);

                // Clear the array that contains neighbours
                neighbours.clear();
            }
        }

        return result;
    }

    Mat convolve(Mat& img, Mat& kernel){

        int rows = img.rows, cols = img.cols, krx = floor(kernel.rows / 2.0), kry = floor(kernel.cols / 2.0);

        // Initialize empty single-channel result image
        Mat result(rows - (krx * 2), cols - (kry * 2), CV_8UC1, Scalar(0));

        int normRes;
        float productSum;
        Scalar kernelSum = sum(kernel);

        // Loop through all the pixels in the image
        for(int x = krx; x < (rows - krx); x++){
            for(int y = kry; y < (cols - kry); y++){

                // Set the sum of products to 0
                productSum = 0.0;

                // Loop through the kernel
                for(int i = -krx; i <= krx; i++){
                    for(int j = -kry; j <= kry; j++){

                        // Get pointers to the kernel and the image values
                        unsigned char *pixelPtr = img.ptr(x + i) + y + j;
                        float *kernelPtr = kernel.ptr<float>(i + krx) + j + kry;

                        // Depointerize the pointers
                        int imgVal = *pixelPtr;
                        float kernelVal = *kernelPtr;

                        // Increment the product sum
                        productSum += (imgVal * kernelVal);
                    }
                }

                if(kernelSum[0] != 0){
                    normRes = abs(productSum) / kernelSum[0];
                }
                else{
                   normRes = abs(productSum);
                }

                result.at<uchar>(x - krx, y - kry) = normRes;
            }
        }

        return result;
    }

    Mat sobelEdge(Mat& img){

        // Get horizontal sobel kernel
        Mat kernel = sobelHorizontal();

        // Find horizontal edges
        Mat horizontalEdge = convolve(img, kernel);

        // Get vertical sobel kernel
        kernel = sobelVertical();

        // Find vertical edges
        Mat verticalEdge = convolve(img, kernel);

        return verticalEdge + horizontalEdge;
    }

    Mat gaussianBlur(Mat& img, int radius, float strenght){

        Mat gaus = gaussianKernel(radius, strenght);

        return convolve(img, gaus);

    }

    Mat backwardMapping(Mat& img, Mat& trans, int scalarX, int scalarY){

        // Get number of rows and columns in the input image
        int rows = img.rows, cols = img.cols;
        int newImgRows = rows * scalarX, newImgCols = cols * scalarY;

        // Create output image with 2x the size
        Mat result(newImgRows, newImgCols, CV_8UC1, Scalar(255));

        // Get the inverse of transform matrix
        Mat inverseTransform = trans.inv();

        // Values for computing
        Vec3f pos;
        Mat oldPos;
        float oldPosX, oldPosY, x1, x2, newVal;
        int upperX, lowerX, upperY, lowerY;

        // Loop through all the pixels in the image
        for(int x = 0; x < newImgRows; x++){
            for(int y = 0; y < newImgCols; y++){
                pos = Vec3f((float)x, (float)y, 1.0);

                // Get the position in the input image
                oldPos = inverseTransform * Mat(pos);

                // Get the position in input image
                oldPosX = oldPos.at<float>(0) / oldPos.at<float>(2);
                oldPosY = oldPos.at<float>(1) / oldPos.at<float>(2);

                // Check if the position is outside the bounds of the input image
                if(oldPosX >= 0 && oldPosY >= 0 && oldPosX < rows && oldPosY < cols){

                    // Get upper and lower bounds of x coordinate
                    upperX = ceil(oldPosX);
                    lowerX = floor(oldPosX);

                    // Get upper and lower bounds of y coordinate
                    upperY = ceil(oldPosY);
                    lowerY = floor(oldPosY);

                    // Bilinear filtering
                    x1 = lerp((float)img.at<uchar>(upperX, lowerY), (float)img.at<uchar>(lowerX, lowerY), upperX - oldPosX);
                    x2 = lerp((float)img.at<uchar>(upperX, upperY), (float)img.at<uchar>(lowerX, upperY), upperX - oldPosX);
                    newVal = lerp(x2, x1, upperY - oldPosY);

                    result.at<uchar>(x, y) = newVal;
                }

            }
        }

        return result;
    }

    /**< Feature extraction */

    Mat harrisCorners(Mat& img, int radius = 2, float k = 0.04, bool norm=false){

        // Get necessary kernels
        Mat gaus = gaussianKernel(radius, 1.0);
        Mat hor = sobelHorizontal();
        Mat vert = sobelVertical();

        // Find gradients in x and y directions
        Mat xGradient = convolve(img, hor);
        Mat yGradient = convolve(img, vert);

        // Get image rows and columns
        int rows = yGradient.rows, cols = yGradient.cols;

        // Sums of gradients
        float Ixx, Iyy, Ixy, det, trace, R;
        Mat result(rows - (radius * 2), cols - (radius * 2), CV_32FC1, Scalar(0));

        // Loop through all the pixels in the image
        for(int x = radius; x < (rows - radius); x++){
            for(int y = radius; y < (cols - radius); y++){

                // Reset sum of gradients
                Ixx = 0.0;
                Iyy = 0.0;
                Ixy = 0.0;

                // Loop through a window
                for(int i = -radius; i <= radius; i++){
                    for(int j = -radius; j <= radius; j++){

                        // Get pointers to the kernel and the image values
                        unsigned char *xPtr = xGradient.ptr(x + i) + y + j;
                        unsigned char *yPtr = yGradient.ptr(x + i) + y + j;

                        // Depointerize the pointers
                        int xVal = *xPtr;
                        int yVal = *yPtr;

                        // Get the weight value for the current window position
                        float *kernelPtr = gaus.ptr<float>(i + radius) + j + radius;
                        float weight = *kernelPtr;

                        // Sum up the gradients
                        Ixx += (weight * (xVal * xVal));
                        Iyy += (weight * (yVal * yVal));
                        Ixy += (weight * (xVal * yVal));
                    }
                }

                det = (Ixx * Iyy) - (Ixy * Ixy);
                trace = Ixx + Iyy;
                R = det - (k * (trace * trace));
                result.at<float>(x - radius, y - radius) = R;
            }
        }

        if (norm){
            // Normalize the values
            double min, max;
            minMaxIdx(result, &min, &max);
            result = (result - min) / (max - min);
        }

        return result;
    }

    Mat shiTomasiCorners(Mat& img, int radius = 2, bool norm=false){

        // Get necessary kernels
        Mat gaus = gaussianKernel(radius, 1.0);
        Mat hor = sobelHorizontal();
        Mat vert = sobelVertical();

        // Find gradients in x and y directions
        Mat xGradient = convolve(img, hor);
        Mat yGradient = convolve(img, vert);

        // Get image rows and columns
        int rows = yGradient.rows, cols = yGradient.cols;

        // Sums of gradients
        float Ixx, Iyy, Ixy, R;
        Mat result(rows - (radius * 2), cols - (radius * 2), CV_32FC1, Scalar(0));

        // Loop through all the pixels in the image
        for(int x = radius; x < (rows - radius); x++){
            for(int y = radius; y < (cols - radius); y++){

                // Reset sum of gradients
                Ixx = 0.0;
                Iyy = 0.0;
                Ixy = 0.0;

                // Loop through a window
                for(int i = -radius; i <= radius; i++){
                    for(int j = -radius; j <= radius; j++){

                        // Get pointers to the kernel and the image values
                        unsigned char *xPtr = xGradient.ptr(x + i) + y + j;
                        unsigned char *yPtr = yGradient.ptr(x + i) + y + j;

                        // Depointerize the pointers
                        int xVal = *xPtr;
                        int yVal = *yPtr;

                        // Get the weight value for the current window position
                        float *kernelPtr = gaus.ptr<float>(i + radius) + j + radius;
                        float weight = *kernelPtr;

                        // Sum up the gradients
                        Ixx += (weight * (xVal * xVal));
                        Iyy += (weight * (yVal * yVal));
                        Ixy += (weight * (xVal * yVal));
                    }
                }

                // Build Structure Matrix
                Mat StructureMat(2, 2, CV_32FC1, Scalar(0));
                StructureMat.at<float>(0, 0) = Ixx;
                StructureMat.at<float>(0, 1) = Ixy;
                StructureMat.at<float>(1, 0) = Ixy;
                StructureMat.at<float>(1, 1) = Iyy;

                // Find the eigenvalues
                cv::Mat eigens;
                cv::eigen(StructureMat, eigens);

                R = std::min(eigens.at<float>(0), eigens.at<float>(1));
                result.at<float>(x - radius, y - radius) = R;
            }
        }

        if (norm){
            // Normalize the values
            double min, max;
            minMaxIdx(result, &min, &max);
            result = (result - min) / (max - min);
        }

        return result;
    }

    void drawCorners(Mat& inImg, Mat& outImg, float r, cv::Scalar color, int markerType = cv::MARKER_CROSS){
        // Get min and max values
        double minimum, maximum;
        cv::minMaxIdx(inImg, &minimum, &maximum);

        // Threshold the response values
        Mat corners;
        cv::threshold(inImg, corners, (maximum * r), 255, 0);

        // Convert to unsinged int
        Mat result;
        corners.convertTo(result, CV_8UC1);

        // Loop through all the pixels in the image
        for(int x = 0; x < result.rows; x++){
            for(int y = 0; y < result.cols; y++){

                unsigned char *pixelPtr = result.ptr(x) + y;
                int val = *pixelPtr;

                if (val > 0){
                    cv::drawMarker(outImg, cv::Point(y + 6, x + 6), color, markerType, 10);
                }

            }
        }


    }
    /**< Histogram processing and operations */

    void histogramStretch(Mat& img){

        // Get number of channels of the input image
        int channels = img.channels();

        // Check if input image is a colored image
        if(channels != 1){
            cout << "Error, expected a grayscale image..." << endl;
        }

        // Get largest and smallest pixel values in an image
        double minimum, maximum;
        minMaxLoc(img, &minimum, &maximum);

        // Calculate the difference between min and max
        int diff = maximum - minimum;

        // Loop through all the pixels in the image
        for(int x = 0; x < img.rows; x++){
            for(int y = 0; y < img.cols; y++){

                // Get pixel value
                unsigned char *pixelPtr = img.ptr(x) + y;
                int pixelVal = *pixelPtr;

                // Perform histogram stretching
                int newPixelVal = ((pixelVal - minimum) / diff) * 255;
                *pixelPtr = newPixelVal;
            }
        }

    }
}

#endif // IMGPROCESSFUNCS_H
