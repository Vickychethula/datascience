#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as snf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[24]:


#read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[25]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","NPG"])
cars.head()


# ### Description of columns 
#   - MPG : Milege of the car(Mile per Gallon)( This is Y-column to be predicted)
#   - HP  : Horse Power of the car (X1 column)
#   - VOL : Volume of the car (size)(X2 column)
#   - SP  : Top speed of the car(Miles per hour)(X3 column)
#   - WT  : Weight of the car(Pounds)(X4 column) 

# In[26]:


# Rearrange the column
cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### EDA 

# In[27]:


cars.info()


# In[28]:


#check for missing values
cars.isna().sum()


# #### Observvation about info(),missing values
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - The data types of the columns are also relevent and valid 

# ##### Prediction Equation:
# $$ 
# \hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4
# $$
# #### MOdel Equation:
# $$
# Y = \hat{Y} + error
# $$ 

# In[29]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="HP", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="HP", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# #### Observation 
# - it is a right skewd plot
# - no.of outliers are 7 

# In[30]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="SP", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="SP", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# ### Observation 
# - it is a right skewd plot
# - no.of outliers are 6

# In[31]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="VOL", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="VOL", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# ## Observation
# - the outlier are the nature of the data it has outliers on both right and left side
# - no of outliers are 2

# In[32]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="WT", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="WT", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# In[33]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="MPG", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="MPG", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# ### Observations from boxplot and Histograms
# - there are some extreme values(Outliers) observed in towards the right tail of SP and HP distribution.
# - In VOL and WT columns a few outliers are observed in both tails of therir distribution
# - The extreme values of car data may have come from the specially designed nature of cars
# - As this is multi-dimension data, the outlier with respect to spatial dimension may have to be considered while building the regresson model 

# ### Checking for duplicated rows

# In[34]:


cars[cars.duplicated()]


# ## Pair plots and Correlation Coefficients

# In[40]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[36]:


cars.corr(numeric_only=True)


# In[37]:


cars.corr()


# ### Observation 
# 1. In the correlation the values of the HP are having higher values than the WT
# 2. Between x and y,all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# 3. Therfore this datasets qualifies for building multiple linear regression model to predict MPG
# 4. Among x columns (x1,x2,x3 and x4), some very high correlation strength are observed between SP vs HP, VOL vs WT
# 5. The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# # preparing a perliminary model considering all x columns

# In[38]:


#Build model
import stats


# In[43]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[44]:


model1.summary()


# oservation from model summary
# - THE R-squared and adjusted R-squared values are good and about 75% of variability in y is explaning by Xcolumns
# - THE probablity value with respect to f-statstics is close to zero ,indicating that all or someof X columns are significant
# - THE p-values for VOL and WT are higher that 5% indicating some interaction issue among themselves,which need to be futer explanation

# performance metrics for model1

# In[41]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[42]:


pred_y1 = model1.predict(cars.iloc[:0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[ ]:




