
# Image transformations

## Backward Mapping

## Transformations

### Scaling

### Rotation

### Shearing

| `Original` | `Sheared` |
| :---:| :---:|
|![fruits](https://user-images.githubusercontent.com/50104866/178017490-d4207355-a50b-48c8-b19d-da5953ecdc4d.jpg)| ![Sheared](https://user-images.githubusercontent.com/50104866/178017526-11a6ac4d-c25d-4846-b97f-8ad9a6e0b8e2.png)|

```C++
    // Convert the image to grayscale
    Mat gray = CVAlg::grayscale(original);

    // Stretch the image's histogram
    CVAlg::histogramStretch(gray);
    
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
    Mat gray = CVAlg::grayscale(original);

    // Stretch the image's histogram
    CVAlg::histogramStretch(gray);

    // Blur the image
    Mat blurred = CVAlg::gausBlur(gray, 7, 2.0);
```

# Sobel edge detection

| `Original` | `Edges` |
| :---:| :---:|
|![board](https://user-images.githubusercontent.com/50104866/178017910-103673e6-cc3c-477e-ae6c-ac110038d985.jpg) | ![Edge](https://user-images.githubusercontent.com/50104866/178017881-9e448181-dcc5-4944-b497-37436b781de1.png)|

```C++
    // Convert the image to grayscale
    Mat gray = CVAlg::grayscale(original);

    // Stretch the image's histogram
    CVAlg::histogramStretch(gray);

    // Extract edges
    Mat edges = CVAlg::sobelEdge(gray);
```

# Features

## Harris Corners Detector

## Shi-Tomasi Corners Detector
