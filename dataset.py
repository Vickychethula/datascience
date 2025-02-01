#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[5]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[6]:


data1.info()


# In[7]:


data1.isnull()


# In[13]:


data1.isnull().sum()


# In[9]:


data1.describe()


# In[10]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[14]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# # observations
# .There are no missing values
# .The daily column values appers to be right-skewed
# .The Sunday column values appears to be right-skewed
# .There are two outliers in both daily column and also in sunday column as observed  from the

# ## scatter plot and correlation strength

# In[21]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0,max(x) + 100)
plt.ylim(0,max(y) + 100)
plt.show()


# In[22]:


data1["daily"].corr(data1["sunday"])


# In[25]:


data1[["daily","sunday"]].corr()


# data1.corr

# In[26]:


import statsmodels.formula.api as snf
model1 = snf.ols("sunday~daily",data = data1).fit()


# In[27]:


model1.summary()


# In[ ]:




