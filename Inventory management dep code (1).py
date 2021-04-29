#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data1=pd.read_csv(r"C:\Users\Admin\Downloads\prorevenue.csv")


# In[3]:


data2=pd.read_csv(r"C:\Users\Admin\Downloads\productdetails.csv")


# In[4]:


data1.head()


# In[5]:


data2.head()


# In[6]:


data1.loc[data1["Generic Holiday"] == 'a',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == 'b',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == 'c',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == '0',"Generic Holiday"] = 0


# In[7]:


data1["Generic Holiday"]= data1["Generic Holiday"].astype(int)


# In[8]:


data1["Generic Holiday"].unique()


# In[9]:


merged = pd.merge(data1, data2, how='outer',
                  left_on='Product type', right_on='product type')
merged = merged.drop('Unnamed: 0', 1) # drop duplicate info


# In[10]:


merged.isnull().sum()


# # Feature Engineering

# In[11]:


merged["No_of_sales"] = (merged["Revenue"] / merged["cost per unit"])*1.10


# In[12]:


merged["No_of_sales"] = merged["No_of_sales"].astype(int)


# In[13]:


merged.head()


# In[14]:


merged.groupby(merged["DayOfWeek"]).No_of_sales.agg(["min",  "max",   "sum",   "count", "mean"])
                                              
                                            
                                             
                                            


# In[15]:


merged = merged.drop("No of purchases", 1)


# In[16]:


merged = merged.drop("cost per unit", 1)


# In[17]:


merged = merged.drop("Time for delivery", 1)


# In[18]:


merged = merged.drop("Education Holiday", 1)


# In[19]:


merged = merged.drop("store status", 1)


# In[20]:


merged = merged.drop("Revenue", 1)


# In[21]:


merged.head()


# In[22]:


X = merged.loc[:, ['product type',"Promotion applied","Generic Holiday",
        'DayOfWeek']]

X.head()


# In[23]:


y = merged.iloc[:, 4]
y.head()


# In[24]:


y.shape


# In[25]:


X.dtypes


# # Normalizing the data 

# In[26]:


from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


# In[27]:


from sklearn.preprocessing import MinMaxScaler


# In[28]:


scaler=MinMaxScaler()


# In[29]:


scaler.fit(X)


# In[30]:


scaled_data = pd.DataFrame(scaler.transform(X),columns=X.columns)


# In[31]:


scaled_data.head()


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.2, random_state = 42)


# In[33]:


#preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model


# In[34]:


# Preparing model                  
ml1 = smf.ols('y~scaled_data',data=merged).fit() # regression model


# In[35]:


# Summary
ml1.summary()


# # Grdient Boosting Regressor 

# In[36]:


# ## Random Forest Regressor

# In[46]:


from sklearn.ensemble import RandomForestRegressor 


# In[47]:


regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)


# In[48]:


regressor.fit(X_train, y_train) 


# In[49]:


# Predicting the values using the model


# In[50]:


Y_pred1 = regressor.predict(X_train) 


# In[51]:


Y_pred2 = regressor.predict(X_test) 


# In[52]:


# plot predicted data 
plt.plot(y_test,Y_pred2,'bo')  
plt.title('Random Forest Regression') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[53]:


# RMSE values


# In[54]:
from sklearn import metrics


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_train, Y_pred1)))


# In[55]:


print('RMSE of testing data:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred2)))


# In[57]:


import pickle
with open('rf_model_inventory.pkl','wb') as f:
    pickle.dump(regressor,f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




