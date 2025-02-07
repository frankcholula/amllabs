import numpy as np

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
class Retangle():
    def __init__(self, length: int, width: int):
        self.length = length
        self.width = width

    def get_area(self):
        return self.length * self.width

def replace_elements_greater_than(arr, threshold, replacement):
    arr = np.array(arr)
    arr[arr > threshold] = replacement
    return arr


if __name__ == "__main__":
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
