import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

source = "../data"


# loads images from ../data for each of the folders
# applies augmentations to each original image (ending in 00000.jpg)
# saves augmented images in the same folder with an added index
def load_images(source_path, destination_path, person_filename):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    cap = cv2.VideoCapture(source_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        filename = f"{frame_count:05d}"
        frame_count += 1
        for i in range(10):
            augmentations = augment_images(frame)
            save_augmented_images(augmentations, destination_path, filename, i)

    cap.release()



# Random horizontal flip based on probability
def random_horizontal_flip(image, probability):
    if np.random.rand() < probability:
        return ImageOps.mirror(image)
    return image


# Random rotation based on max angle
def random_rotation(image, max_angle):
    rotate_angle = np.random.uniform(-max_angle, max_angle)
    return image.rotate(rotate_angle)


# Random perspective distortion using PIL
def random_perspective(image, distortion_scale=0.05):
    width, height = image.size
    # generation of random shift values for each corner of image
    # these values are calculated as a fraction of the image's dimensions
    # and scaled by the distortion scale
    left = np.random.randint(-int(distortion_scale * width), int(distortion_scale * width))
    top = np.random.randint(-int(distortion_scale * height), int(distortion_scale * height))
    right = np.random.randint(-int(distortion_scale * width), int(distortion_scale * width))
    bottom = np.random.randint(-int(distortion_scale * height), int(distortion_scale * height))

    # defined original and final coordinates for the four corners of the image
    # original coordinates - top-left, top-right, bottom-left, bottom-right
    # final coordinates are modified by the random shift values calculated above
    original_coords = [(0, 0), (width, 0), (0, height), (width, height)]
    shifted_coords = [(left, top), (width - right, top), (left, height - bottom), (width - right, height - bottom)]

    # calculates the transformation coefficients using the original and shifted coordinates
    # these coefficients are then used to transform the image and build the final perspective distorted image
    coefficients = _find_coefficients(shifted_coords, original_coords)

    # apply perspective transformation to the image
    # image.PERSPECTIVE - needs coefficients and a method of interpolation (BICUBIC - usually very smooth)
    return image.transform((width, height), Image.PERSPECTIVE, coefficients, Image.BICUBIC)


def _find_coefficients(pa, pb):
    # finds coefficients by solving a linear system of equations,
    # which map the points in pa (original) to the points in pb (shifted)
    matrix = []
    for p1, p2 in zip(pa, pb):
        # for each pair of points (original and destination), build equations for the matrix
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    # converting the matrix to a numpy array and solving the system of equations
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    # the solution gives the coefficients
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def custom_color_jitter(image, brightness=0.25, contrast=0.25, saturation=0.25):
    # similar to torchvisions ColorJitter
    # random brightness change
    if brightness > 0:
        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

    # random contrast change
    if contrast > 0:
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    # random saturation change
    if saturation > 0:
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)

    return image


# use augmentations to create a list of augmented images
def augment_images(image):
    resized_image = image.resize((256, 256), Image.BICUBIC)

    aug_image = custom_color_jitter(resized_image, np.random.uniform(0, 0.15), np.random.uniform(0, 0.15),
                                    np.random.uniform(0, 0.15))
    aug_image = random_horizontal_flip(aug_image, 0.3)
    aug_image = random_rotation(aug_image, 5)
    aug_image = random_perspective(aug_image)

    return aug_image


def save_augmented_images(image, base_path, filename, index):
    augmentation_index = f"{index:05d}"  # 5 digits, zero-padded for augmented images
    new_filename = f'{filename}_{augmentation_index}.jpg' # original_index = index of original image
    image.save(os.path.join(base_path, new_filename))


if __name__ == '__main__':
    source_folder = f"../data/videos/"
    for filename in os.listdir(source_folder):
        filename = os.path.basename(filename).split('.')[0]
        destination_folder = f"../data/{filename}"
        source_file = f"{source_folder}{filename}.mp4"
        load_images(source_file, destination_folder, filename)
