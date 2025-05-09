# Self-Checkout Kiosk for Fruits using AI and IoT

This project presents a smart, AI-powered self-checkout system designed to automate the billing process for fresh produce in supermarkets. Using computer vision and machine learning, the kiosk identifies fruits placed under a camera, calculates the price based on weight, and generates a billing interface — minimizing manual labor and queue times.

## Features

- Real-time fruit identification using a CNN-based image classification model (MobileNetV2).
- Flask web app for image upload and prediction.
- USB camera integration for live fruit detection (`live_predict.py`).
- HTML/CSS-based user interface with prediction display.
- Easily integrable with weighing scale hardware (IoT).
- Custom-trained model (`fruit_model_mobilenet.h5`) on real fruit images.

---

## Project Structure

| File                       | Description |
|----------------------------|-------------|
| `app.py`                   | Flask web server to serve the model and web interface. |
| `fruit_model_mobilenet.h5` | Trained MobileNetV2 model for fruit classification. |
| `index.html`               | Front-end HTML file for user interaction. |
| `styles.css`               | CSS file for styling the web page. |
| `live_predict.py`          | Script to capture live image from USB camera and predict the fruit. |
| `self_check1.ipynb`        | Jupyter notebook used for model training and evaluation. |

---

## Setup Instructions

1. Download the zip file and extract it in your pc
2. Open VS Code and open the folder.
3. Install required packages
   
   pip install tensorflow flask opencv-python numpy

4. Run the flask app

   python app.py

5. Access it on 127.0.0.1:5000 (in the terminal)
6. To test with live camera

   python live_predict.py



Demo

Upload or capture a fruit image.

The system predicts the fruit name with confidence.

Can be extended to weigh the fruit and calculate total price.



---

Technologies Used

Python

TensorFlow / Keras

OpenCV

Flask

HTML5 / CSS3

MobileNetV2 (Transfer Learning)



---

Future Scope

Integration with weighing scale hardware (Arduino/Raspberry Pi).

Add subtype classification (e.g., different types of mangoes).

Full self-service kiosk with touchscreen and billing system.

Deploy model on edge devices (Jetson Nano / Raspberry Pi).


Author

Shivin Pruthi
