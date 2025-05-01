from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("fruit_model_mobilenet.h5")
train_dir = r"C:\Users\pruth\Downloads\VIT Downloads\Self_Checkout\archive\MY_data\train"
class_labels = sorted(os.listdir(train_dir))

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 150)) / 255.0  
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["file"]
        img = preprocess_image(image)
        prediction = model.predict(img)
        print("Raw Predictions:", prediction)
        print("Predicted Label:", class_labels[np.argmax(prediction)])
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        return f"Predicted: {class_labels[class_index]}, Confidence: {confidence:.2f}"
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

