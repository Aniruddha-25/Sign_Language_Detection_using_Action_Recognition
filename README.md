# Sign Language Detection using Action Recognition

## Description
### This project implements a sign language detection system using action recognition techniques. It leverages computer vision and deep learning to identify and classify sign language gestures from Camera input, enabling real-time translation of sign language into text.

## Features
- Real-Time Sign Language Gesture Detection
- Action Recognition using Deep Learning Models
- User-friendly Interface for Camera Input
- Support for Multiple Sign Language Gestures
- Easy to Extend for Additional Gestures

## Technologies Used
- Python
- OpenCV
- TensorFlow / PyTorch (specify which one you use)
- NumPy
- Streamlit (if you have a web interface)
- Other relevant libraries

## How to Use?
1. Clone the repository:
3. Install the required dependencies:
   
    ````
    pip install -r requirements.txt
    ````


5. Run the application:
   
   To Create a Training Model.
   Note :- Run This Program if there is no model file in  the folder name ````models\ASL.h5````
   ````
   python src/train_asl_model.py
   ````
   
   Run the Program to Detect American Sign Language
   ````
   python src\detect_signs.py
   ````

7. Follow the on-screen instructions to start detecting sign language gestures from your webcam 
