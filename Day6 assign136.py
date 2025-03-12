#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import the pandas library
import pandas as pd

# Create a dictionary containing the data
data = {
    'Name': ['John', 'Alice', 'Bob', 'Diana'],
    'Age': [28, 34, 23, 29],
    'Department': ['HR', 'IT', 'Marketing', 'Finance'],
    'Salary': [45000, 60000, 35000, 50000]
}

# Create a DataFrame using the dictionary
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


# In[2]:


# Import the pandas library
import pandas as pd

# Create the DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob', 'Diana'],
    'Age': [28, 34, 23, 29],
    'Department': ['HR', 'IT', 'Marketing', 'Finance'],
    'Salary': [45000, 60000, 35000, 50000]
}

df = pd.DataFrame(data)

# Task 1: Display the first 2 rows of the DataFrame
print("First 2 rows of the DataFrame:")
print(df.head(2))

# Task 2: Add a new column named 'Bonus' where the bonus is 10% of the salary
df['Bonus'] = df['Salary'] * 0.10

# Display the DataFrame after adding the Bonus column
print("\nDataFrame after adding 'Bonus' column:")
print(df)

# Task 3: Calculate the average salary of employees in the DataFrame
average_salary = df['Salary'].mean()
print("\nAverage salary of employees:", average_salary)

# Task 4: Filter and display employees who are older than 25
older_than_25 = df[df['Age'] > 25]
print("\nEmployees older than 25:")
print(older_than_25)


# In[ ]:




