#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Take input for the number n
n = int(input("Enter a positive integer n: "))

# Initialize a variable to store the sum
sum_of_even_numbers = 0

# Loop through all numbers from 1 to n
for num in range(1, n + 1):
    if num % 2 == 0:
        sum_of_even_numbers += num

# Print the result
print("The sum of all even numbers between 1 and", n, "is:", sum_of_even_numbers)


# In[ ]:




