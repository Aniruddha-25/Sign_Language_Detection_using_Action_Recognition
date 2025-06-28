import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ASL.h5")
model = load_model(model_path)

# Class labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

def preprocess_image(image, target_size=(96, 96)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

cap = cv2.VideoCapture(0)
word = ""
capture_word = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Region of interest for hand sign
    roi = frame[100:300, 100:300]
    processed_roi = preprocess_image(roi)

    # Predict sign
    prediction = model.predict(processed_roi, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Build output word
    if capture_word:
        if predicted_label == 'del':
            word = word[:-1]
        elif predicted_label == 'space':
            word += ' '
        elif predicted_label != 'nothing':
            word += predicted_label
        capture_word = False

    cv2.putText(frame, word, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Hand Sign Detection', frame)

    # Keyboard controls: 'q' to quit, 's' to add letter
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        capture_word = True

cap.release()
cv2.destroyAllWindows()
