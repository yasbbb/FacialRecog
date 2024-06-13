## Facial Recognition Application 

Developed a real-time facial recognition application using OpenCV joined with Flask, capable of recognizing and differentiating up to two simultaneous users with a ~95% accuracy rate.

![Screenshot 2024-06-12 at 10 29 38 PM](https://github.com/yashbbb009/FacialRecog/assets/165434548/0364b0dc-fcb1-4364-b7c6-4713da9e50e7)

This project is the development of a web application that integrates real-time facial recognition capabilities using Flask, a micro web framework in Python, for handling user interactions and OpenCV for computer vision tasks. The application allows users to create training datasets by capturing face images, trains a face recognition model using LBPH, and recognizes and differentiates individuals in real-time by streaming video frames to the client from a webcame connected to your computer.


Installation
ensure you have the following installed
- python
- opencv
- flask
- numpy 

Execution 
install project locally using npm to run 

    $ pip install opencv-contrib-python
    $ pip install numpy
    $ pip install flask
    $ cd WebApplication_FaceRecognition/
    $ export FLASK_APP=client2
    $ python -m flask run
