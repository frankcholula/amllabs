import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)
import cv2
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Question 1
def sum_of_numbers():
    number = int(input("Enter a positive number: "))
    if number < 0:
        logging.info("Please enter a positive number")
    sum = 0  # Initialize sum variable
    for i in range(1, number + 1):
        sum += i  # Add i instead of 1
    return sum


# Question 2
class Retangle:
    def __init__(self, length: int, width: int):
        self.length = length
        self.width = width

    def get_area(self):
        return self.length * self.width


# Question 3
def replace_elements_greater_than(arr: np.array, threshold: int, replacement: int):
    arr = np.array(arr)
    arr[arr > threshold] = replacement
    return arr


# Question 4
def visualize_sine_cosine():
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label="Sine")
    plt.plot(x, y2, label="Cosine")
    plt.legend()
    plt.show()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info(f"{method.__name__} took {te - ts:2.4f} sec")
        return result

    return timed


# Question 5
@timeit
def process_img_sklearn(greyscale: np.array, patch_size: int = 4):
    patches = extract_patches_2d(greyscale, (patch_size, patch_size))
    logging.debug(f"Patches shape: {patches.shape}")
    patch_means = patches.mean(axis=(1, 2))
    # use broadcasting
    new_patches = np.full(
        (patches.shape[0], patch_size, patch_size),
        patch_means[:, np.newaxis, np.newaxis],
    )
    logging.debug(f"New patches shape: {new_patches.shape}")
    averaged = reconstruct_from_patches_2d(new_patches, greyscale.shape)
    return averaged


@timeit
def process_img_slow(greyscale: np.array, patch_size: int = 4):
    h, w = greyscale.shape
    averaged = np.zeros_like(greyscale)
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = greyscale[i : i + patch_size, j : j + patch_size]
            averaged[i : i + patch_size, j : j + patch_size] = np.mean(patch)
    return averaged


@timeit
def process_img_fast(greyscale: np.array, patch_size: int = 4):
    h, w = greyscale.shape
    new_h = h - (h % patch_size)
    new_w = w - (w % patch_size)
    greyscale = greyscale[:new_h, :new_w]
    patches = greyscale.reshape(
        new_h // patch_size, patch_size, new_w // patch_size, patch_size
    )
    patch_means = patches.mean(axis=(1, 3))
    # Expand the reduced array back to the original size
    averaged = np.repeat(np.repeat(patch_means, patch_size, axis=0), patch_size, axis=1)
    return averaged


def average_color_of_img(img_path: str, patch_size: int = 4):
    try:
        img = cv2.imread(img_path)
        logging.debug(f"Image shape: {img.shape}")
        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logging.debug(f"Greyscale shape: {greyscale.shape}")
        averaged_fast = process_img_sklearn(greyscale, patch_size)
        averaged_slow = process_img_slow(greyscale, patch_size)
        averaged_sklearn = process_img_fast(greyscale, patch_size)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(greyscale, cmap="gray")
        ax1.set_title("Original")
        ax2.imshow(averaged_fast, cmap="gray")
        ax2.set_title("Fast Processing")
        ax3.imshow(averaged_slow, cmap="gray")
        ax3.set_title("Slow Processing")
        ax4.imshow(averaged_sklearn, cmap="gray")
        ax4.set_title("Sklearn Processing (Slowest)")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    # Exercise 5
    average_color_of_img("./surrey.png")

    # Exercise 4
    visualize_sine_cosine()

    # Exercise 3
    arr = np.array([12, 34, 56, 78, 90])
    threshold = 50
    replacement = -1
    logging.info(f"Array before replacement: {arr}")
    arr = replace_elements_greater_than(arr, threshold, replacement)
    logging.info(f"Array after replacement: {arr}")

    # Exercise 2
    rect = Retangle(5, 10)
    logging.info(f"Area of rectangle: {rect.get_area()}")

    # Exercise 1
    logging.info(f"Sum of numbers: {sum_of_numbers()}")
