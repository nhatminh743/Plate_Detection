import keras
import cv2
import tensorflow as tf
import numpy as np

# Load model
model_dir = r'/home/minhpn/Desktop/Green_Parking/CNN_Recognize/CNN_Model.keras'
model = keras.models.load_model(model_dir)

# Read image
image_path = '/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Letter/0229_05817_b_plate/ROI_0.jpg'
image = cv2.imread(image_path)

# Preprocess image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Optional but good practice for TensorFlow
image = tf.image.resize(image, (12, 28))
image = image / 255.0  # Normalize to [0, 1]
image = image.numpy().astype(np.float32)

# Add batch dimension
input_batch = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(input_batch)
print("Raw prediction:", prediction)

# Get predicted class
ALPHA = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
         'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
         'U', 'V', 'X', 'Y', 'Z']

predicted_index = np.argmax(prediction)
predicted_label = ALPHA[predicted_index]
confidence = np.max(prediction)

print(f"Predicted Class: {predicted_label} (Index: {predicted_index}, Confidence: {confidence:.2f})")