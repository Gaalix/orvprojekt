import unittest
from unittest import mock
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import keras_tuner as kt
import model.model as mm

class TestModel(unittest.TestCase):

    @mock.patch('tensorflow.keras.applications.MobileNetV2')
    @mock.patch('tensorflow.keras.optimizers.Adam')
    @mock.patch('tensorflow.keras.models.Model')
    def model_building(self, mock_model, mock_adam, mock_mobilenet):
        mock_mobilenet.return_value = mock.Mock()
        mock_adam.return_value = mock.Mock()
        mock_model.return_value = mock.Mock()
        hp = kt.HyperParameters()
        mm.build_model(hp)
        mock_mobilenet.assert_called_once()
        mock_adam.assert_called_once()
        mock_model.assert_called_once()

    @mock.patch('keras.src.legacy.preprocessing.image.ImageDataGenerator')
    def data_loading(self, mock_datagen):
        mock_datagen.return_value = mock.Mock()
        mm.load_data('data_dir')
        mock_datagen.assert_called_once()

    @mock.patch('keras_tuner.BayesianOptimization')
    @mock.patch('tensorflow.keras.callbacks.EarlyStopping')
    def hyperparameters_tuning(self, mock_early_stopping, mock_bayesian_optimization):
        mock_early_stopping.return_value = mock.Mock()
        mock_bayesian_optimization.return_value = mock.Mock()
        mm.tune_hyperparameters('train_generator', 'val_generator')
        mock_early_stopping.assert_called_once()
        mock_bayesian_optimization.assert_called_once()

if __name__ == '__main__':
    unittest.main()