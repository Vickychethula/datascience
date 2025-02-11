#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


print(type(Univ))
print(Univ.shape)
print(Univ.size)


# In[5]:


Univ.describe()


# In[6]:


# Read all numeric columns in to Univ1

Univ1 = Univ.iloc[:,1:]


# In[7]:


Univ1


# In[8]:


cols = Univ1.columns


# In[9]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scaled_Univ_df = pd.DataFrame(scalar.fit_transform(Univ1), columns=Univ1.columns)
scaled_Univ_df


# In[10]:


# Build 3 clusters using Kmeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3,random_state=0)#Specify 3 clusters
clusters_new.fit(scaled_Univ_df)


# In[11]:


#PRINT the cluster label
clusters_new.labels_


# In[12]:


set(clusters_new.labels_)


# In[13]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[14]:


Univ


# In[15]:


Univ.sort_values(by = "clusterid_new")


# In[16]:


#use grouby() to find aggregatted(mean)values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# # observation
# - cluster2 appears to be the top rated universities cluster as the cut off score,top,SFRatio parameters mean values are highest
# - cluster 1 appears to occupy the middle level rated universities
# - cluster 0 comes as the lowest level rated universities

# In[19]:


wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:





# In[ ]:




