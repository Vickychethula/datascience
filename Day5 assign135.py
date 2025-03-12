#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Task 1: Ask the user to enter a positive integer n
n = int(input("Enter a positive integer n: "))

# Task 2: Use a for loop to print all numbers from 1 to n on separate lines
print("Numbers from 1 to", n, "using a for loop:")
for num in range(1, n + 1):
    print(num)

# Task 3: Use a while loop to calculate the sum of all numbers from 1 to n and print the result
sum_of_numbers = 0
i = 1
while i <= n:
    sum_of_numbers += i
    i += 1

print("The sum of all numbers from 1 to", n, "is:", sum_of_numbers)


# In[3]:


# Task 1: Define a function to calculate the square of a number
def calculate_square(n):
    return n ** 2

# Task 2: Ask the user to input a positive integer
n = int(input("Enter a positive integer: "))

# Task 3: Call the function and display the result
square = calculate_square(n)
print(f"The square of {n} is: {square}")


# In[ ]:




