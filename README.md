Face Mask Detection

This repository contains a face-mask detector built using Tensorflow/keras and OpenCV

I have built a classifier whick classifies whether a image contains mask or not. The classifier is built using MobileNetV2 pre trained model. I have used OpenCV for real time video capture. For face detection I have used pre-trained caffemodel. For face detection you can refer my another repository, click [here](https://github.com/nehal2000/face-detection-dnn.git)

I have already trained the classifier and saved the model in model directory. The model was trained by executing dnn_model.py

The mask_detector.py loads the trained model and uses OpenCV for real time video capture and feeds the frames to the model and predicts the output

To run the face detection module execute " python mask_detector.py " in your terminal or command prompt. Press esc to stop the program.

REQUIREMENTS:
1. Tensorflow
2. imutils
3. OpenCV
4. sklearn
