import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from keras.src.utils import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
import os

def create_and_train_user_model(is_user, framer_dir):
    images = []
    labels = []

    for file in os.listdir(framer_dir):
        file_path = os.path.join(framer_dir, file)
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
        labels.append(is_user)

    images = np.array(images) / 255.0
    labels = np.array(labels)

    labels_categorical = tf.keras.utils.to_categorical(labels, num_classes=2)

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(images, labels_categorical, batch_size=32,
              epochs=30,
              callbacks=[early_stopping])

    model_save_path = f"user_models/model_{user_id}.keras"
    model.save(model_save_path)
    return model_save_path

if __name__ == "__main__":
    user_id = 103
    is_user = 1
    framer_dir = "../data/Gal"
    model_path = create_and_train_user_model(is_user, framer_dir)


