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


# In[18]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[21]:


#detection of outliers in the columns


# In[22]:


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


# In[23]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')


# In[24]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[25]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert= False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ###using mu +/-3*sigma limits(standard deviation method

# In[26]:


data1["Ozone"].describe()


# In[27]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# Observations

# its is observed thatonly two outliers are defined using std method
# in boxplot method more no of outliers are identified
# this is because the assumption of nomality is not satisfied in the column

# In[28]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[31]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin plot")


# In[33]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[34]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange", palette="Set1", size=6, jitter = True)


# In[35]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[36]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[37]:


#correlation coeffiecient and pair plots


# In[38]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[40]:


# compute pearson correlation coefficient
# between wind speed and temperature
data1["Wind"].corr(data1["Temp"])


# In[42]:


# read all numeric columns into a new table
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




