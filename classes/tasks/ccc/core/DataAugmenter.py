import math
import random
from typing import Tuple

import cv2
import numpy as np

from functional.image_processing import rgb_to_bgr


class DataAugmenter:

    def __init__(self, train_size: Tuple = (512, 512), angle: int = 60, scale: Tuple = (0.1, 1.0), color: float = 0.8):
        self._train_size = train_size

        # Rotation angle
        self._angle = angle

        # Patch scale
        self._scale = scale

        # Color rescaling
        self._color = color

    def __get_random_angle(self) -> float:
        return (random.random() - 0.5) * self._angle

    @staticmethod
    def __rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle (in degrees).
        The returned image will be large enough to hold the entire new image, with a black background
        """

        # Get the image size (note: NumPy stores image matrices backwards)
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        rot_mat_no_translate = np.matrix(rot_mat[0:2, 0:2])

        image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_no_translate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_no_translate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_no_translate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos, x_neg = [x for x in x_coords if x > 0], [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos, y_neg = [y for y in y_coords if y > 0], [y for y in y_coords if y < 0]

        right_bound, left_bound, top_bound, bot_bound = max(x_pos), min(x_neg), max(y_pos), min(y_neg)
        new_w, new_h = int(abs(right_bound - left_bound)), int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

        # Compute the transform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def __largest_rotated_rect(w: float, h: float, angle: float) -> Tuple:
        """
        Given a rectangle of size w x h that has been rotated by 'angle' (in radians), computes the width and height of
        the largest possible axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow. Converted to Python by Aaron Snoswell
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        length = h if (w < h) else w
        d = length * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return bb_w - 2 * x, bb_h - 2 * y

    def __crop_around_center(self, image: np.ndarray, width: float, height: float) -> np.ndarray:
        """ Given a NumPy / OpenCV 2 image, crops it to the given width and height around it's centre point """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        width = image_size[0] if width > image_size[0] else width
        height = image_size[1] if height > image_size[1] else height

        x1, x2 = int(image_center[0] - width * 0.5), int(image_center[0] + width * 0.5)
        y1, y2 = int(image_center[1] - height * 0.5), int(image_center[1] + height * 0.5)

        return cv2.resize(image[y1:y2, x1:x2], self._train_size)

    def _rotate_and_crop(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        if angle is None:
            angle = self.__get_random_angle()
        width, height = image.shape[:2]
        target_width, target_height = self.__largest_rotated_rect(width, height, math.radians(angle))
        return self.__crop_around_center(self.__rotate_image(image, angle), target_width, target_height)

    @staticmethod
    def _random_flip(img: np.ndarray, p: int = None) -> np.ndarray:
        """
        Perform random left/right flip with probability p (defaults to 0.5)
        @param img: the image to be flipped
        @param p: the probability according with image should be flipped (in {0, 1})
        @return: the flipped image with probability p, else the original image
        """
        if p is None:
            p = random.randint(0, 1)
        return img[:, ::-1].astype(np.float32) if p else img.astype(np.float32)

    def __get_random_scale(self, img: np.ndarray) -> float:
        scale = math.exp(random.random() * math.log(self._scale[1] / self._scale[0])) * self._scale[0]
        return min(max(int(round(min(img.shape[:2]) * scale)), 10), min(img.shape[:2]))

    def _rescale(self, img: np.ndarray, scale: float = None) -> np.ndarray:
        if scale is None:
            scale = self.__get_random_scale(img)
        start_x = random.randrange(0, img.shape[0] - scale + 1)
        start_y = random.randrange(0, img.shape[1] - scale + 1)
        return img[start_x:start_x + scale, start_y:start_y + scale]

    def augment(self, img: np.ndarray, illumination: np.ndarray) -> Tuple:
        img = self._rescale(img)
        img = self._rotate_and_crop(img)
        img = self._random_flip(img)

        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * self._color - 0.5 * self._color
        img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]], dtype=np.float32)
        new_image = np.clip(img, 0, 65535)

        new_illuminant = np.zeros_like(illumination)
        illumination = rgb_to_bgr(illumination)
        for i in range(3):
            for j in range(3):
                new_illuminant[i] += illumination[j] * color_aug[i, j]
        new_illuminant = rgb_to_bgr(np.clip(new_illuminant, 0.01, 100))

        return new_image, new_illuminant
