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


# In[6]:


data2.head()


# In[31]:


data1.loc[data1["Generic Holiday"] == 'a',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == 'b',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == 'c',"Generic Holiday"] = 1
data1.loc[data1["Generic Holiday"] == '0',"Generic Holiday"] = 0


# In[33]:


data1["Generic Holiday"]= data1["Generic Holiday"].astype(int)


# In[34]:


data1["Generic Holiday"].unique()


# In[35]:


merged = pd.merge(data1, data2, how='outer',
                  left_on='Product type', right_on='product type')
merged = merged.drop('Unnamed: 0', 1) # drop duplicate info


# In[36]:


merged.isnull().sum()


# # Feature Engineering

# In[37]:


merged["No_of_sales"] = (merged["Revenue"] / merged["cost per unit"])*1.10


# In[38]:


merged["No_of_sales"] = merged["No_of_sales"].astype(int)


# In[39]:


merged.head()


# In[40]:


merged.groupby(merged["DayOfWeek"]).No_of_sales.agg(["min",  "max",   "sum",   "count", "mean"])
                                              
                                            
                                             
                                            


# In[41]:


merged = merged.drop("No of purchases", 1)


# In[42]:


merged = merged.drop("cost per unit", 1)


# In[43]:


merged = merged.drop("Time for delivery", 1)


# In[44]:


merged = merged.drop("Education Holiday", 1)


# In[45]:


merged = merged.drop("store status", 1)


# In[46]:


merged = merged.drop("Revenue", 1)


# In[47]:


merged.head()


# In[48]:


X = merged.loc[:, ['product type',"Promotion applied","Generic Holiday",
        'DayOfWeek']]

X.head()


# In[49]:


y = merged.iloc[:, 4]
y.head()


# In[50]:


y.shape


# In[51]:


X.dtypes


# # Normalizing the data 

# In[52]:


from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


# In[53]:


from sklearn.preprocessing import MinMaxScaler


# In[54]:


scaler=MinMaxScaler()


# In[55]:


scaler.fit(X)


# In[57]:


scaled_data = pd.DataFrame(scaler.transform(X),columns=X.columns)


# In[58]:


scaled_data.head()


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.2, random_state = 42)


# In[61]:


#preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model


# In[64]:


# Preparing model                  
ml1 = smf.ols('y~scaled_data',data=merged).fit() # regression model


# In[65]:


# Summary
ml1.summary()


# # Grdient Boosting Regressor 

# In[68]:


from sklearn.ensemble import GradientBoostingRegressor


# In[69]:


from sklearn.ensemble import GradientBoostingRegressor


# In[70]:


params={"n_estimators":150,"max_depth":4,"learning_rate":1,"criterion":"mse"}
gradient_boosting_regressor_model=GradientBoostingRegressor(**params)


# In[71]:


gradient_boosting_regressor_model.fit(X_train,y_train)


# In[72]:


PRED=gradient_boosting_regressor_model.predict(X_test)


# In[74]:


from sklearn import metrics


# In[75]:


metrics.r2_score(y_test,PRED)


# In[76]:


print(gradient_boosting_regressor_model.score(X_train,y_train))


# In[77]:


print('MAE:', metrics.mean_absolute_error(y_test,PRED))
print('MSE:', metrics.mean_squared_error(y_test,PRED))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,PRED)))


# In[79]:


import pickle
with open('BTACH_PICLKE','wb') as f:
    pickle.dump(gradient_boosting_regressor_model,f)


# In[ ]:




