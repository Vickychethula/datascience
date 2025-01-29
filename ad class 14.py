#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1.info()


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1[data1.duplicated()]


# In[10]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[11]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[12]:


data1.isnull().sum()


# In[13]:


cols = data1.columns
colors = ['black','yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("mean of Ozone: ",mean_ozone)


# In[15]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


data1['Ozone'] = data1['Ozone'].fillna(mean_ozone)
data1.isnull().sum()


# In[17]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("mean of Solar: ",mean_solar)


# In[18]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[19]:


data1['Solar'] = data1['Solar'].fillna(mean_ozone)
data1.isnull().sum()


# In[29]:


data1["Month"].median()
mean_month = data1["Month"].mean()
print("Median of month: ", median_month)
print("Mean of month: ", mean_month)


# In[30]:


data1["Month"] = data1["Month"].fillna(median_month)
data1.isnull().sum()


# In[31]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[32]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[33]:


fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1,3]})


# In[26]:


fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title('Boxplot')
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1['Ozone'],kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")



# In[34]:


plt.tight_layout()


# In[28]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# ### method 2

# In[21]:


data1["Ozone"].describe()


# In[22]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x<(mu - 3*sigma))or (x > (mu + 3*sigma))):
        print(x)


# In[23]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# #### other visualisation to understand the graph
# 

# In[37]:


#create a figure for violin plot

sns.violinplot(data=data1["Ozone"],color="lightpink")
plt.title("Violin plot")

#show the plot
plt.show()


# In[47]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone",color="orange" ,palette="Set1", size=6)


# In[48]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set1", size=6)


# In[51]:


sns.kdeplot(data=data1["Ozone"], color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[52]:


sns.boxplot(data = data1, x ="Weather", y="Ozone")


# ### corelation cofficent and pair plots
# 

# In[53]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[54]:


# compute pearson correlation coefficient
data1["Wind"].corr(data1["Temp"])


# In[58]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




