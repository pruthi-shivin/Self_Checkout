import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model("fruit_model_mobilenet.h5")

class_labels = sorted(os.listdir("archive/My_data/train"))

IMG_SIZE = (150, 150)

def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

cap = cv2.VideoCapture(0)

print("📷 Press SPACE to capture and predict. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera - Press SPACE to Predict", frame)

    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == 32:  
        img = preprocess(frame)
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        predicted_label = class_labels[class_idx]
        print(f"🟢 Predicted: {predicted_label}, Confidence: {confidence:.2f}")

        display_text = f"{predicted_label} ({confidence:.2f})"
        result_frame = cv2.putText(frame.copy(), display_text, (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction Result", result_frame)
        cv2.waitKey(2000)  

cap.release()
cv2.destroyAllWindows()
