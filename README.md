# CV Algorithms
A set of naive implementations of popular CV algorithms implemented in `C++` using `OpenCV` library. The examples below contain code snippets showcasing example use of algorithms.

# Table of contents
- [Image-thresholding](#Image-thresholding)
- [Image Transformations](#Image-Transformations)
- [Gaussian Blurring](#Gaussian-Blurring)
- [Sobel edge detection](#Sobel-edge-detection)
- [Features](#Features)

# Image thresholding

| `Original` | `Binary` |
| :---:| :---:|
|![sudoku](https://user-images.githubusercontent.com/50104866/178046409-b5905053-e777-494c-ab34-3fb9c5c5eeb7.png) | ![Binary](https://user-images.githubusercontent.com/50104866/178046386-30209dc4-c40d-4f6c-a880-d9c2837442ac.png)|

```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);

    //Binarize image
    Mat binary = CVAlg::threshold(gray, 125);
```

# Image Transformations

## Scaling

| `Original` | `Scaled` |
| :---:| :---:|
|![butterfly](https://user-images.githubusercontent.com/50104866/178106450-10805a9f-6d99-47d2-9d5d-290d72418c4d.jpg)|![Enlarged](https://user-images.githubusercontent.com/50104866/178106462-62149a57-f3be-4724-9c1b-96b617b00435.png)|


```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);

    // Scale the image
    Mat scale = CVAlg::scaleMat(2.0, 2.0);
    Mat result = CVAlg::backwardMapping(gray, scale, 2, 2);
```

## Rotation about Z axis

| `Original` | `Scaled` |
| :---:| :---:|

```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);

    // Scale the image
    Mat scale = CVAlg::scaleMat(2.0, 2.0);
    Mat result = CVAlg::backwardMapping(gray, scale, 2, 2);
```

## Shearing

| `Original` | `Sheared` |
| :---:| :---:|
|![fruits](https://user-images.githubusercontent.com/50104866/178017490-d4207355-a50b-48c8-b19d-da5953ecdc4d.jpg)| ![Sheared](https://user-images.githubusercontent.com/50104866/178017526-11a6ac4d-c25d-4846-b97f-8ad9a6e0b8e2.png)|

```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);
    
    // Perform Shearing
    Mat shear = CVAlg::shearMat(0.0, 1.1);
    Mat sheared = CVAlg::backwardMapping(gray, shear, 1, 2.5);

    // Resize image
    Mat result;
    cv::resize(sheared, result, cv::Size(), 0.5, 0.5);
```

# Gaussian Blurring

| `Original` | `Blurred` |
| :---:| :---:|
|![messi5](https://user-images.githubusercontent.com/50104866/178017771-976034df-095e-477a-8e84-259ccf8e4cb9.jpg)|  ![Blurred](https://user-images.githubusercontent.com/50104866/178017811-e9bf8d02-aada-4778-ad09-95cba2b1948b.png)|

```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);
    
    // Blur the image
    Mat blurred = CVAlg::gausBlur(gray, 7, 2.0);
```

# Sobel edge detection

| `Original` | `Edges` |
| :---:| :---:|
|![board](https://user-images.githubusercontent.com/50104866/178017910-103673e6-cc3c-477e-ae6c-ac110038d985.jpg) | ![Edge](https://user-images.githubusercontent.com/50104866/178017881-9e448181-dcc5-4944-b497-37436b781de1.png)|

```C++
    // Convert the image to grayscale
    Mat gray(original.rows, original.cols, CV_8UC1, Scalar(0));
    CVAlg::grayscale(original, gray);

    // Extract edges
    Mat edges = CVAlg::sobelEdge(gray);
```

# Features

## Harris Corners Detector
![Corners](https://user-images.githubusercontent.com/50104866/178045575-cb38ad40-e82f-4d83-a00b-e96bb9f48200.png)


```C++
    // Convert the image to grayscale
    Mat gray = CVAlg::grayscale(original);

    // Extract corners
    Mat corners = CVAlg::harrisCorners(gray, 7, 0.04);

    // Draw corners
    CVAlg::drawCorners(corners, original, 0.5, cv::Scalar(0, 0, 255), cv::MARKER_DIAMOND);
```


## Shi-Tomasi Corners Detector
![Corners](https://user-images.githubusercontent.com/50104866/178031086-d5072d23-7ddb-4e18-9662-78e1ff081c7f.png)

```C++
    // Convert the image to grayscale
    Mat gray = CVAlg::grayscale(original);

    // Extract corners
    Mat corners = CVAlg::shiTomasiCorners(gray, 5);

    // Draw corners
    CVAlg::drawCorners(corners, original, 0.65, cv::Scalar(0, 0, 255));
```
