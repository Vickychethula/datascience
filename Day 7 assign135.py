#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("Day_7_sales_data.csv")
data


# In[4]:


import pandas as pd

# Load the data
data = pd.read_csv('Day_7_sales_data.csv')

# Display the first 5 rows of the dataset
print(data.head())

# Print basic statistics of the numerical columns
print(data.describe())


# In[5]:


# Calculate the total sales for each region
total_sales_by_region = data.groupby('Region')['Sales'].sum()

# Display the total sales for each region
print(total_sales_by_region)


# In[ ]:




