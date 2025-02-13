#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install mixed library
get_ipython().system('pip install mlxtend')


# In[4]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[ ]:


#from google.colab import files
#uploaded = files.upload()


# In[6]:


#print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[7]:


titanic.info()


# # observation
# - there are no null values
# - all columns are object and categorised
# - as the columns are catergorical,we can adopt one-hot-encoding
# - all women and adult are survived
# - all child and men are not survived

# In[12]:


titanic['Class'].value_counts()


# In[11]:


#plot a bar chart to visualize the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[13]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# adults travelled more than the children

# In[14]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# surived rate farless than notsurvived

# In[15]:


#perform onehot encoding on categorical columns
df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[16]:


df.info()


# # apriori algorithm

# In[17]:


#apply Apriori algorithm to get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True)
frequent_itemsets


# In[18]:


frequent_itemsets.info()


# In[19]:


#Generate asssociation rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[20]:


rules.sort_values(by='lift', ascending = True)


# In[ ]:


#rules[rules[]]


# In[22]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




