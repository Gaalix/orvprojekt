import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import keras_tuner as kt


# Function for building the model
def build_model(hp):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for i in range(hp.Int('num_layers', 1, 3)):
        x = Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu')(x)
        x = Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


# Function for loading the data
def load_data(data_dir):
    datagen = ImageDataGenerator(validation_split=0.2)

    train_generator = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=100, class_mode='binary',
                                                  subset='training')
    val_generator = datagen.flow_from_directory(data_dir, target_size=(224, 224), batch_size=100, class_mode='binary',
                                                subset='validation')

    print(train_generator.class_indices)

    return train_generator, val_generator


# Hyperparameter tuning
def tune_hyperparameters(train_generator, val_generator):
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        directory='my_dir',
        project_name='2')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

    tuner.search(train_generator, validation_data=val_generator, epochs=2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Najboljši hiperparametri so:
    - Število slojev: {best_hps.get('num_layers')}
    - Število nevronov v vsakem sloju: {[best_hps.get('units_' + str(i)) for i in range(best_hps.get('num_layers'))]}
    - Stopnja učenja: {best_hps.get('learning_rate')}
    """)

    return tuner.hypermodel.build(best_hps)


data_dir = "../data"
train_generator, val_generator = load_data(data_dir)

best_model = tune_hyperparameters(train_generator, val_generator)
best_model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[EarlyStopping(patience=3)])

user_id = 2
model_path = f"user_models/model_{user_id}.keras"
best_model.save(model_path)
