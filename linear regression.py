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


plt.scatter(data1["daily"],data1["sunday"])


# In[8]:


data1.describe()


# In[9]:


data1.sum()


# ## Q.what is the linear regression
# - linear reggresion is the method of finding or obtaining an mathematical eq between x ansd y and fit a st.line
# ## Q.how can we say it is a linear reggresion problem
# - if the y colum is continous float then it is linear regression is it si categorial then it si a classification 
# ## Q.which library is used in l.r
# - we use statsmodels.formula.api another is sklearn presently statsmodel
# - It is simple due to sinlge line# 

# In[10]:


data1.isnull().sum()


# In[11]:


data1.describe()


# In[12]:


# Boxplot for daily column 
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[13]:


# Boxplot for sunday column 
plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[14]:


# hist plot for daily sales
sns.histplot(data1['daily'], kde = True, stat = 'density')
plt.show()


# In[15]:


# hist plot for sunday sales
sns.histplot(data1['sunday'], kde = True, stat = 'density')
plt.show()


# ### Observations 
# - there are ni missing values
# - the sdaily column values appear to be right-skwed
# - the sunday column values also appear to be right-skwed
# - there are two outliers in both daily column and also in sunday column as observes from the above boxplot# 

# ### Scatter plot and Correlation Strength 

# In[16]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.xlabel("Daily")
plt.ylabel("Sunday")
plt.show()


# In[17]:


data1["daily"].corr(data1["sunday"])


# In[18]:


data1[["daily","sunday"]].corr()


# In[19]:


data1.corr(numeric_only=True)


# ### correlation coefficient formula
# r = $$
#     \frac{\sum_{i=0}^n (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=0}^n(x_i - \overline{x})^2 \sum_{i=0}^n(y_i - \overline{y})^2}}
#     $$

# #### Observation 
# - the relationship between a(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - the correlation is strong and positive with Pearson's correlation coefficient of 0.958154

# ## Fit a linear Regression Model

# In[21]:


# Build regression model 
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[22]:


model1.summary()


# ### (03/02/2025)
# ## model.summary() parameters: 
# - in regression analysis using libearies like stsmodel in python  

# ### Interpretation:
# - $$
#   R^2 = 1-Perfect fit (all variance explained).
#   $$
# - $$
#   R^2 = 0 : Model does not explian any variance.
#   $$
# - $$
#   R^2 close to 1 -> good model fit.
#   $$
# - $$
#   R^2 close to 0 -> Poor model fit.
#   $$

# In[23]:


#PLOt THE SCATTER PLOT AND OVERLAY THE FITTED STRAIGHT LINE USING MATPLOTLIB
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y,color = "m", marker = "o", s = 30)
b0 = 13.84
b1= 1.33
# predivates response vector 
y_hat = b0 +b1*x
# plotting the regression line 
plt.plot(x, y_hat, color = "g")

# putting labels 
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# ### Observations from Model Summary:
# - the probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - therefore the intercept coefficient may not be that much significant in prediction
# - However the p-value for "daily"{beta_1} is 0.00 < 0.05
# - Therefore the beta_1 co-efficient is highly significant and is contributint to prediction
# - p-value should be 0.05(probability)

# In[24]:


model1.params


# In[25]:


# Print the most 
print(f'model t-values:\n{model1.tvalues}\n---------------\nmodel p-values: \n{model1.pvalues}')


# In[26]:


# Print teh Quality of fitted line (R squared values)
model1.rsquared,model1.rsquared_adj


# ### Predit for new data point 
# - Equation of prediction is
#   $$
#   \hat{Y} = 13.84 + 1.33X
#   $$ 

# In[27]:


# Prdict for 200 and 300 daily circulation 
newdata = pd.Series([200,300,1500])


# In[28]:


data_pred = pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[29]:


model1.predict(data_pred)


# In[30]:


# Predicate on all given traingn data
pred = model1.predict(data1["daily"])
pred


# In[31]:


# Add predicate values as a column in data1
data1["Y_hat"] = pred
data1


# In[32]:


# Compute the error values (residuals) and add as another column 
data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1 # it is done to find mean square and root mean square 


# In[33]:


# Compute Mean Squared Error for the model

mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# ### (04/02/20205)
# ### Assumptions in linear simple linear regression 
# 1. **Linearity:** The relaationship between the predictors and the response is linear.
# 2. **Independence:** Observations are independent of each other.
# 3. **Homoscedasticity:** The residuals (Y - Y_hat) exhibit constant variance at all levels of the predictors.
# 4. **Normal Distribution of Errors:** The residuals(errors) of the model are normally distributed.

# ### Checking the model residuals scatter plot (for homoscadasticity)

# In[34]:


# Plot the residuals versus y_hat (to check whether residuals are independent of each 
plt.scatter(data1["Y_hat"], data1["residuals"])


# ### OBSERVATIONS:
# - there appears to be no treand and the residuals are randomly placed around the zero error linr
# - Hence the assumption of Homoscedasticity (constan variance in residuals) is satisfied
# - Data points are randomly scattered.zero error line

# In[35]:


# Plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm 
sm.qqplot(data1["residuals"],line ='45',fit=True)
plt.show()


# In[39]:


# Plot the kde distribution for residuals 
sns.histplot(data1["residuals"])


# In[41]:


## Observations 
- The data points are seen to closely follows the reference line of normality.
- Hence the residuals are approximatly normally distributed as also can be seen from the kde distribution.
- All the assumpitons are satisfactory hence the model is performing well


# In[ ]:




