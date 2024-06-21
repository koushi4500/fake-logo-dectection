import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model("M:/ml projects/Externship-Fake-Real-Logo-Detection/scripts/output/logo_detection_model.h5")

# Example: Load a new image for prediction
image_path = r"M:\ml projects\Externship-Fake-Real-Logo-Detection\dataset\test\Genuine\000002_ef67f5045e3a44bdb9001d956746a391.jpg"
# Replace with your image path
img = load_img(image_path, target_size=(128, 128))  # Assuming your model expects (128, 128) images
img_array = img_to_array(img)
img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
X_new = np.array([img_array])  # Adjust for batch size if needed

# Make predictions
predictions = model.predict(X_new)
print(predictions)
