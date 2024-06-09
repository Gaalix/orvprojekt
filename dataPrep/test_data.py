import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import os
import cv2
import dataPrep.dataManipulation as dm

class TestDataManipulation(unittest.TestCase):

    def test_random_horizontal_flip_does_not_flip_image_when_probability_low(self):
        image = Image.new('RGB', (60, 30), color = 'red')
        flipped_image = dm.random_horizontal_flip(image, 0)
        self.assertEqual(image.tobytes(), flipped_image.tobytes())

    def test_random_rotation_rotates_image(self):
        image = Image.new('RGB', (60, 30), color = 'red')
        rotated_image = dm.random_rotation(image, 45)
        self.assertNotEqual(image.tobytes(), rotated_image.tobytes())

    def test_random_perspective_distorts_image(self):
        image = Image.new('RGB', (60, 30), color = 'red')
        distorted_image = dm.random_perspective(image, 0.5)
        self.assertNotEqual(image.tobytes(), distorted_image.tobytes())

    def test_custom_color_jitter_changes_image(self):
        image = Image.new('RGB', (60, 30), color = 'red')
        jittered_image = dm.custom_color_jitter(image, 1, 1, 1)
        self.assertNotEqual(image.tobytes(), jittered_image.tobytes())

    def test_augment_images_returns_augmented_image(self):
        image = Image.new('RGB', (60, 30), color = 'red')
        augmented_image = dm.augment_images(image)
        self.assertNotEqual(image.tobytes(), augmented_image.tobytes())

    @patch('os.path.join')
    @patch('PIL.Image.Image.save')
    def test_save_augmented_images_saves_image(self, mock_save, mock_join):
        image = Image.new('RGB', (60, 30), color = 'red')
        dm.save_augmented_images(image, 'base_path', 'filename', 1)
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()