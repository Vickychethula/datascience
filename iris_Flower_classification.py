#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[4]:


iris.info()


# In[5]:


iris.describe()


# In[7]:


#print all duplicated rows
iris[iris.duplicated(keep = False)]


# In[8]:


#drop duplicated rows
iris.drop_duplicates(keep='first', inplace = True)
iris


# In[9]:


iris.isnull().sum()


# #### Observations

# * There are 150 rows and 5 columns
# * There are no null values
# * There is one duplicated row (101 and 142 are duplicated)
# * The x-columns are sepal.length,sepal,petal.length and petal.width
# * All the x-columns are continuous
# * The y-column is "variety"(object) which is to be converted to categorical(to be predict)
# * There are three flower categories(classes)

# #### Transform the y-column to categorical using LabelEncoder()

# In[13]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris


# In[ ]:




