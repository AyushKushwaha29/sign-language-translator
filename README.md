This project is a real-time Sign Language Translator that leverages Core Machine Learning and Image Processing techniques to bridge the communication gap between the hearing and speech-impaired community and the general public.

The system uses TensorFlow with a MobileNetV2-based transfer learning model to accurately classify American Sign Language (ASL) hand gestures. By integrating MediaPipe and OpenCV, the project performs efficient hand detection and preprocessing, ensuring robust performance in live environments.

A Flask-based web server combined with Flask-SocketIO enables real-time gesture recognition and communication. The recognized signs are then translated into text and speech using Deep Translator and pyttsx3, making the system interactive and inclusive.

🔑 Key Features:

Deep Learning Model: MobileNetV2 with transfer learning for high-accuracy ASL gesture classification.

Real-Time Processing: MediaPipe + OpenCV for hand landmark detection and preprocessing.

Web Interface: Flask + Flask-SocketIO for live streaming and gesture-to-text translation.

Speech & Translation: pyttsx3 for text-to-speech output, Deep Translator for multi-language support.

Cross-Platform: Works seamlessly across browsers and devices with real-time performance.

⚙️ Tech Stack:

Machine Learning: TensorFlow, Keras, NumPy, h5py

Image Processing: MediaPipe, OpenCV, Pillow

Backend & Communication: Flask, Flask-SocketIO, Flask-CORS, Eventlet

Text-to-Speech & Translation: pyttsx3, Deep Translator

🎯 Impact:

This project aims to enhance inclusivity by providing a practical tool that helps break communication barriers. It enables real-time interaction between sign language users and non-sign language speakers, making daily conversations, education, and accessibility more seamless.
