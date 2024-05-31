from keras.src.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf

def test_model_with_image(model_path, image_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)

    # Preprocess the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the prediction
    predicted_class = np.argmax(prediction)

    return predicted_class

# Test the function
model_path = "user_models/model_1.keras"

# loop for testing different images
for i in range(1, 9):
    image_path = f"../data/Tomaz/00000_0000{i}.jpg"
    print(test_model_with_image(model_path, image_path))