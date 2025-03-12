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


# In[9]:


data1.rename({'solar.R': 'solar'}, axis=1, inplace = True)
data1


# ##impute the missing value

# In[10]:


data1.info()


# In[11]:


#display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[12]:


#visualize the data1 missing values using heat map
cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[13]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[14]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[15]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of solar.r: ", median_solar)
print("Mean of solar.r: ", mean_solar)


# In[16]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[21]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[20]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[22]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[23]:


#detection of outliers in the columns


# In[25]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1,3]})
sns.boxplot(data=data1["Ozone"],ax=axes[0], color='skyblue',width=0.5, orient = 'h' )
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[ ]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')

