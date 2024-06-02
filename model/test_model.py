import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def load_test_data(test_data_dir):
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(test_data_dir, target_size=(224, 224), batch_size=100, class_mode='binary')
    return test_generator

def test_model(model_path, test_data_dir):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load the test data
    test_generator = load_test_data(test_data_dir)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)

    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

# Usage
test_model("user_models/model_2.keras", "../testData/")

#0 = user
#1 = unknown