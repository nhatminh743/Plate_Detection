import keras
import cv2
import numpy as np

# Load model
model_dir = r'/home/minhpn/Desktop/Green_Parking/CNN_Recognize/CNN_Model.keras'
model = keras.models.load_model(model_dir)

# Read image
image_path = '/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Letter/0230_01270_b_plate/ROI_0.jpg'
image = cv2.imread(image_path)

# Preprocess image (Normalization)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (28, 12))  # (width, height) for OpenCV
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Add batch dimension
input_batch = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(input_batch)
print("Raw prediction:", prediction)

# Decode prediction
ALPHA = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
         'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
         'U', 'V', 'X', 'Y', 'Z']

predicted_index = np.argmax(prediction)
predicted_label = ALPHA[predicted_index]
confidence = np.max(prediction)

print(f"Predicted Class: {predicted_label} (Index: {predicted_index}, Confidence: {confidence:.2f})")
