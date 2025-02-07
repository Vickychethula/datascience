#!/usr/bin/env python
# coding: utf-8

# ##### (04/02/2025)
# ### Multi LInear Regression   
# - Multi linear regression(multiple linear regression)

# ### Assumptions:
# 1. **Linearity:** the relationship between the predicators(X) and the response(Y) is linear.
# 2. **Independence:** Observations are independent of each other.
# 3. **Homoscedasticity:** The residuals(Y-Y_hat) exhibots variance at all levels of the predictor.
# 4. **Normal Distribution of Error:** The residuals of the model are normally distributed.
# 5. **No Multicollinearity:** The independent variables should not be too highly correlated with each other.
# - Voilations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# ### Description of columns 
#   - MPG : Milege of the car(Mile per Gallon)( This is Y-column to be predicted)
#   - HP  : Horse Power of the car (X1 column)
#   - VOL : Volume of the car (size)(X2 column)
#   - SP  : Top speed of the car(Miles per hour)(X3 column)
#   - WT  : Weight of the car(Pounds)(X4 column)

# In[3]:


# Rearrange the column
cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### EDA

# In[4]:


cars.info()


# In[5]:


#check for missing values
cars.isna().sum()


# #### Observvation about info(),missing values
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - The data types of the columns are also relevent and valid

# ###### (05/02/2025)
# 

# #### Prediction Equation:
# $$ 
# \hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4
# $$
# #### MOdel Equation:
# $$
# Y = \hat{Y} + error
# $$

# In[6]:


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

# In[7]:


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


# #### Observation 
# - it is a right skewd plot
# - no.of outliers are 6

# In[8]:


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


# ### Observation
# - the outlier are the nature of the data it has outliers on both right and left side
# - no of outliers are 2

# In[9]:


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


# In[10]:


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

# In[11]:


cars[cars.duplicated()]


# ### Pair plots and Correlation Coefficients

# In[12]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr(numeric_only=True)


# In[14]:


cars.corr()


# #### (06/02/2025)

# ## Observations ##
# * Between x and y all the x variables are showing moderate to high correlation strengths ,highest being between HP and MPH
# * therefore the dataset qualifies for building a multiple linear regression model to predict MPG
# * Among x columns (x1,x2,x3,and x4),some very high correlation strengths are observed between SP vs HP,VOL vs WT
# * The high correlation among x columns is not desirable as it might lead to multicollinearly problem 

# ## Preparing a premilinarynmodel considering all x columns ##

# In[15]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[16]:


model1.summary()


# ## Observations from model summary ##
# * The R-squared and adjusted R-squared values are good and qbout 75% of variability in Y is explained by X columns
# * The probability value with respect to F-statistoic is close to zero,indicating that all or someof X columns are significant
# * The P-values for VOL and WT are higher than 5% indicating issue among themselves,which need to be future explored

# ## Performance metrices for model1 ##

# In[17]:


# Find the performance metrices
# Create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[18]:


# Predict for the given X  data columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# checking for multicollinearity among X-columns using VIF method

# In[20]:


#cars.head()


# In[21]:


#compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 


# In[22]:


# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# observations of VIF values:
#     -the idea range of VIF values shall be between 0 to 10.however slight higher values can be tolerated
#     -Hence it is decided to drop one of the column(either VOL or WT)

# In[23]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[25]:


#bulid model2 on cars dataset
import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[26]:


model2.summary()


# performance metrics for model2
# 

# In[28]:


#Find the performance metrics

df2 = pd.DataFrame()
df2["actual_y2"]= cars["MPG"]
df2.head()


# In[29]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[37]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# In[ ]:





# In[ ]:




