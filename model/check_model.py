import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import base64
import os

# Define the path to your model
MODEL_PATH = os.path.join(os.getcwd(), "orvprojekt\\model\\user_models", "model_tomaz.keras")

app = Flask(__name__)

def process_image(image_data):
    image_data = base64.b64decode(image_data)
    image = tf.image.decode_image(image_data, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debugging: Check if model path is correct and model exists
        print("Model path:", MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        print("Model exists:", os.path.exists(MODEL_PATH))
        
        image_data = request.json.get('image', None)
        
        if image_data is None:
            raise ValueError("No image data provided")
        
        # Debugging: Check the received image data length
        print("Received image data length:", len(image_data))
        
        processed_image = process_image(image_data)
        
        # Debugging: Check processed image shape
        print("Processed image shape:", processed_image.shape)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Debugging: Check if the model was loaded successfully
        print("Model loaded successfully")
        
        prediction = model.predict(processed_image)
        
        # Debugging: Check the prediction result
        print("Prediction result:", prediction)
        
        result = bool(prediction[0][0] > 0.5)
        return jsonify({'result': result})
    except Exception as e:
        error_message = str(e)
        print("Error:", error_message)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    app.run(debug=True, host='0.0.0.0')
