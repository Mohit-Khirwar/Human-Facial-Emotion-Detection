# Human Facial Expression Detection

This project demonstrates **human facial expression detection** using a fine-tuned **MobileNetV2** model, trained on the **FER2023** dataset. The system uses **Haar Cascade** for detecting faces in images and video streams, followed by the classification of facial expressions into one of seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Overview

The core idea behind this project is to enable detection and recognition of human facial expressions through a lightweight yet powerful neural network model. The project is implemented using a fine-tuned version of **MobileNetV2**—a highly efficient architecture well-suited for devices with limited computational resources. The face detection is handled by the robust **Haar Cascade** algorithm from OpenCV.

In addition to the command-line interface, the project also includes a **Streamlit web application**, providing an easy-to-use interface for testing and interacting with the model.

## Web Application

Test out the facial expression detection model directly in your browser through the interactive **Streamlit web app**:

[**Try the Web App Here**](https://human-facial-emotion-detection.streamlit.app/)

The web app allows users to upload images to perform facial expression analysis. It showcases the model's performance in identifying emotions from human faces.

## Features

- **Haar Cascade Face Detection**: Efficient face detection for images and video streams.
- **MobileNetV2-based Emotion Recognition**: Fine-tuned MobileNetV2 model to classify facial expressions into seven categories.
- **Streamlit Web Interface**: An easy-to-use and visually interactive web application for testing the model.

## Model and Dataset

1. **FER2023 Dataset**:  
   The model is trained on the **FER2023** dataset, which contains a large collection of face images categorized into seven emotion classes:
   - Angry
   - Disgust
   - Fear
   - Happy
   - Sad
   - Surprise
   - Neutral

2. **MobileNetV2 Fine-tuning**:  
   The pre-trained **MobileNetV2** architecture was fine-tuned using FER2023 to improve its performance for facial expression recognition. This ensures a balance between accuracy and computational efficiency, making the model suitable for real-time applications.

3. **Haar Cascade Face Detection**:  
   OpenCV’s Haar Cascade classifier is employed for detecting faces in images or video frames. It ensures that only the detected face regions are passed to the MobileNetV2 model for expression classification.

## Usage

### 1. Image-Based Facial Expression Detection

You can detect facial expressions in static images by providing an image as input, and the system will identify the emotions displayed.

### 2. Streamlit Web App

Access the model's functionality through a simple and intuitive web app. The app enables users to upload images for expression analysis. 


## Future Improvements

- **Enhanced Face Detection**: Explore using more advanced face detection techniques like **MTCNN** or **RetinaFace** for better accuracy in challenging environments.
- **Multiface Detection**: Extend support for detecting and classifying multiple faces in an image or video stream simultaneously.
- **Additional Datasets**: Retrain or fine-tune the model on other facial expression datasets to improve performance across different demographics.

## Acknowledgments

- The **FER2023** dataset for providing a rich collection of labeled images for training the model.
- The **MobileNetV2** architecture for its lightweight and efficient design, enabling real-time predictions.
- The **OpenCV** library for robust face detection with **Haar Cascade**.

