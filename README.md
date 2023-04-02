# AiDriverSafety
This project is an AI-based driver safety system that detects driver fatigue and triggers an alarm to alert the driver to take a break.
AI Driver Safety System
This project is an AI-based driver safety system that detects driver fatigue and triggers an alarm to alert the driver to take a break. The system uses a facial landmark detection algorithm to track the driver's eyes and detect if they are closed for an extended period or if they yawn. If the driver's eyes are closed or they yawn for too long, the system triggers an alarm to alert the driver.

Requirements
Python 3.x
OpenCV 4.x
Dlib 19.x
Playsound
Installation
Clone the repository to your local machine.
Install the required dependencies using the following command: pip install -r requirements.txt.
Usage
Run the main.py file using the following command: python main.py.
The program will open your computer's webcam and start detecting your facial landmarks.
If the system detects that you are yawning or closing your eyes for an extended period, an alarm will be triggered.
Features
Detects driver fatigue based on facial landmarks.
Triggers an alarm to alert the driver.
Easy to use and customize.
Uses OpenCV and Dlib for facial landmark detection.
Playsound library for triggering the alarm.
Known Issues
The system may not work accurately in low-light conditions or if the driver is wearing glasses or a hat that covers their face.
False alarms may be triggered if the driver is not actually fatigued.
License
This project is licensed under the MIT License. Feel free to use and modify the code as per your requirements.

Credits
The project is created by Shubham Singh and inspired by DriverDetection model. Special thanks to Srinivas sir.
