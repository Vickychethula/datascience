#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


#printing the information
data.info()


# In[4]:


#dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[5]:


#drop duplocate column(temp c) and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[8]:


#drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[13]:


data1.rename({'solar.R': 'solar'}, axis=1, inplace = True)
data1


# ##impute the missing value

# In[12]:


data1.info()


# In[15]:


#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[16]:


#visualize the data1 missing values using heat map
cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[26]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[28]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[22]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of solar.r: ", median_solar)
print("Mean of solar.r: ", mean_solar)


# In[29]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_ozone)
data1.isnull().sum()


# In[ ]:




