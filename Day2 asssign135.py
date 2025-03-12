#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create a List with 5 elements
my_list = [10, 20, 30, 40, 50]

# Create a Tuple with 5 elements
my_tuple = (100, 200, 300, 400, 500)

# Create a Dictionary with 5 key-value pairs
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York', 'job': 'Engineer', 'hobby': 'Reading'}

# Accessing elements from the List
print("Accessing elements from List:")
print("Element at index 2 (third element) in List:", my_list[2]) 
print("Element at index -1 (last element) in List:", my_list[-1])  

# Accessing elements from the Tuple
print("\nAccessing elements from Tuple:")
print("Element at index 1 (second element) in Tuple:", my_tuple[1])  
print("Element at index -3 (third last element) in Tuple:", my_tuple[-3])  

# Accessing elements from the Dictionary
print("\nAccessing elements from Dictionary:")
print("Value for the key 'name' in Dictionary:", my_dict['name']) 
print("Value for the key 'hobby' in Dictionary:", my_dict['hobby'])  

# Try accessing using get() method for dictionary
print("\nUsing get() method to access dictionary elements:")
print("Value for the key 'city' in Dictionary using get():", my_dict.get('city'))
print("Value for the key 'job' in Dictionary using get():", my_dict.get('job'))  


# In[ ]:




