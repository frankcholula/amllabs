# Question 1
def sum_of_numbers(number: int):
    sum = 0  # Initialize sum variable
    for i in range(1, number + 1):
        sum += i  # Add i instead of 1
    return sum

# Get user input
if __name__ == "__main__":
    number = int(input("Enter a positive number: "))
    result = sum_of_numbers(number)
    print(f"The sum of numbers from 1 to {number} is: {result}")

