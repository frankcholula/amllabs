import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
import time


# Question 1
def sum_of_numbers():
    number = int(input("Enter a positive number: "))
    if number < 0:
        print("Please enter a positive number")
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
        print(f"{method.__name__} took {te - ts:2.4f} sec")
        return result

    return timed


# Question 5

def average_color_of_img(img_path: str, patch_size: int = 4):
    try:
        img = cv2.imread(img_path)
        print(f"Image shape: {img.shape}")
        greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Greyscale shape: {greyscale.shape}")

        @timeit
        def process_img_fast(greyscale):
            h, w = greyscale.shape
            patches = extract_patches_2d(greyscale, (patch_size, patch_size))
            means = np.mean(patches, axis=(1,2))
            # Print debug info
            print(f"Original shape: {greyscale.shape}")
            print(f"Patches shape: {patches.shape}")
            print(f"Means shape: {means.shape}")
            
            # Calculate correct grid dimensions
            grid_h = (h - patch_size + 1)
            grid_w = (w - patch_size + 1)
            print(f"Grid dimensions: {grid_h}x{grid_w}")
            
            # Reshape with correct dimensions
            means = means.reshape(grid_h, grid_w)
            
            # Expand to original size
            return means
            
        @timeit
        def process_img_slow(greyscale):
            h, w = greyscale.shape
            averaged = np.zeros_like(greyscale)
            for i in range(0, h - patch_size + 1, patch_size):
                for j in range(0, w - patch_size + 1, patch_size):
                    patch = greyscale[i : i + patch_size, j : j + patch_size]
                    averaged[i : i + patch_size, j : j + patch_size] = np.mean(patch)
            return averaged
        
        averaged = process_img_fast(greyscale)

        # Displays the image
        cv2.imshow("Averaged", averaged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Exercise 5
    average_color_of_img("./surrey.png")

    # Exercise 4
    visualize_sine_cosine()

    # Exercise 3
    arr = np.array([12, 34, 56, 78, 90])
    threshold = 50
    replacement = -1
    print(f"Array before replacement: {arr}")
    arr = replace_elements_greater_than(arr, threshold, replacement)
    print(f"Array after replacement: {arr}")

    # Exercise 2
    rect = Retangle(5, 10)
    print(f"Area of rectangle: {rect.get_area()}")

    # Exercise 1
    print(f"Sum of numbers: {sum_of_numbers()}")
