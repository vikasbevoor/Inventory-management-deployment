#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns


# In[113]:





# In[114]:


data1=pd.read_csv(r"C:\Users\Admin\Downloads\prorevenue.csv")


# In[115]:


data2=pd.read_csv(r"C:\Users\Admin\Downloads\productdetails.csv")


# In[116]:


data1.head()


# In[117]:


data1.shape


# In[118]:


data2.head()


# In[119]:


data2.shape


# In[120]:


merged = pd.merge(data1, data2, how='outer',
                  left_on='product type', right_on='product type')
merged = merged.drop('Unnamed: 0', 1) # drop duplicate info
merged.head(20)


# In[121]:


merged.tail(20)


# In[122]:


merged.shape


# In[123]:


merged.isnull().sum()


# In[124]:


print(merged.nunique())


# In[125]:


#Statistical description of the dataset
merged.describe()


# In[126]:


print(merged.iloc[[159220]])


# In[127]:


print(merged.iloc[[1000]])


# In[128]:


# 1. Number of enteries 
print(merged.shape)
# We have 1017209 rows and 8 columns 

# 2. Total number of products & unique values of the columns 
print("*****************")
print(merged.nunique())


# In[129]:


# 3. Count of the historical and active state 
print("*****************")
print(merged[merged['store status'] == 'open']['store status'].count())
print(merged[merged['store status'] == 'close']['store status'].count())


# In[130]:


merged.dtypes


# In[514]:


my_report=sweetviz.analyze([merged,"DATA1"],target_feat="No of purchases")


# In[515]:


my_report.show_html("Report.html")


# In[516]:


data1['store status'].value_counts().plot.bar(title="freq dist store status ")


# In[517]:


data1['Promotion applied'].value_counts().plot.bar(title="Freq dist of File Type")


# In[518]:


data1['Generic Holiday'].value_counts().plot.bar(title="Freq dist of generic holiday")


# In[519]:


data1['Education Holiday'].value_counts().plot.bar(title="Freq dist of File Type")


# In[520]:


data1['DayOfWeek'].value_counts().plot.bar(title="Freq dist dayofweek")


# In[521]:


# No of purchages  vs store status
sns.catplot(y = "No of purchases", x = "store status", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[522]:


# No of purchages vs Education Holiday
sns.catplot(y = "No of purchases", x = "Education Holiday", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[523]:


# No of purchages vs Day of week
sns.catplot(y = "No of purchases", x = "DayOfWeek", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[524]:


# No of purchages vs Promotion applied
sns.catplot(y = "No of purchases", x = "Promotion applied", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[525]:


# No of purchages vs Day of week
sns.catplot(y = "Revenue", x = "DayOfWeek", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[526]:


# No of purchages vs Day of week
sns.catplot(y = "Revenue", x = "Promotion applied", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[527]:


# No of purchages vs Day of week
sns.catplot(y = "Revenue", x = "store status", data = data1.sort_values("product type", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[528]:


# Finds correlation between Independent and dependent attributes
axis = plt.subplots(figsize=(20,14))
sns.heatmap(merged.corr(),annot = True)
plt.show()


# # Revenue and No Of Purchages have very High correlation i,e 0.89%

# # Aggregation of values Based on Day Of Week 

# In[131]:


merged.groupby(merged["DayOfWeek"]).Revenue.agg(["min",
                                               "max",
                                               "sum",
                                               "count",
                                               "mean"])


# # Feature Engineering

# In[132]:


merged["No_of_sales"] = merged["Revenue"] / merged["cost per unit"]
merged.head()


# In[133]:


merged.groupby(merged["DayOfWeek"]).No_of_sales.agg(["min",  "max",   "sum",   "count", "mean"])
                                              
                                            
                                             
                                            


# In[134]:


merged.head()


# In[135]:


merged.groupby(["product type"]).sum().sort_values("No_of_sales", ascending=True).head(10)


# In[136]:


merged = merged.drop("No of purchases", 1)


# In[137]:


merged = merged.drop("cost per unit", 1)


# In[138]:


merged = merged.drop("Time for delivery", 1)


# In[139]:


merged = merged.drop("Education Holiday", 1)


# In[140]:


merged = merged.drop("store status", 1)


# In[141]:


merged = merged.drop("Revenue", 1)


# In[142]:


merged.head()


# In[143]:


merged.head()


# In[144]:


merged.isnull().sum()


# In[145]:


X = merged.loc[:, ['product type',"Promotion applied","Generic Holiday",
        'DayOfWeek']]

X.head()


# In[146]:


y = merged.iloc[:, 4]
y.head()


# In[147]:


y.shape


# In[148]:


X.dtypes


# In[149]:


X.dtypes


# In[150]:


y.dtypes


# # Normalizing the data 

# In[151]:


# plot original distribution plot with larger value feature
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))
ax1.set_title('Original Distributions')

sns.kdeplot(X['product type'], ax=ax1)
sns.kdeplot(X['Promotion applied'], ax=ax1)
sns.kdeplot(X['Generic Holiday'], ax=ax1)
sns.kdeplot(X['DayOfWeek'], ax=ax1);


# In[152]:


from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')


# In[153]:


X


# # Robust scaler

# In[154]:


r_scaler = preprocessing.RobustScaler()
df_r = r_scaler.fit_transform(X)

df_r = pd.DataFrame(df_r,columns=["product type","Promotion applied","Generic Holiday","DayOfWeek"])

fig, (ax2) = plt.subplots(ncols=1, figsize=(10, 8))
ax2.set_title('After RobustScaler')

sns.kdeplot(df_r['product type'], ax=ax2)
sns.kdeplot(df_r['Promotion applied'], ax=ax2)
sns.kdeplot(df_r['Generic Holiday'], ax=ax2)
sns.kdeplot(df_r['DayOfWeek'], ax=ax2);


# In[155]:


df_r


# In[156]:


mins = [df_r[col].min() for col in df_r.columns]
mins


# In[157]:


maxs = [df_r[col].max() for col in df_r.columns]
maxs


# # Combined Plot

# In[158]:


# Combined plot.

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))


ax1.set_title('Original Distributions')

sns.kdeplot(X['product type'], ax=ax1)
sns.kdeplot(X['Promotion applied'], ax=ax1)
sns.kdeplot(X['Generic Holiday'], ax=ax1)
sns.kdeplot(X['DayOfWeek'], ax=ax1);



ax2.set_title('After RobustScaler')

sns.kdeplot(df_r['product type'], ax=ax2)
sns.kdeplot(df_r['Promotion applied'], ax=ax2)
sns.kdeplot(df_r['Generic Holiday'], ax=ax2)
sns.kdeplot(df_r['DayOfWeek'], ax=ax2);


# # VIF Factor Checking for mine variables 

# In[159]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[160]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


# In[161]:


vif


# # For All the features VIF Factor is less than 10 only so we consider 

# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_r, y, test_size = 0.2, random_state = 42)


# In[49]:


X_train


# In[50]:


X_test


# In[51]:


y_train


# In[52]:


y_test


# In[162]:


# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[163]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[164]:


X


# In[165]:


df_r


# In[166]:


#preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model


# In[167]:


# Preparing model                  
ml1 = smf.ols('y~X',data=merged).fit() # regression model


# In[168]:


# Summary
ml1.summary()


# # P Value for all mine input Variables are less Than 0.05 only ,Hypothesis Also correct 

# # Liner Regression 

# In[960]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean


# In[793]:


# Bulding and fitting the Linear Regression model
linearModel = LinearRegression()
linearModel.fit(X_train, y_train)
  
# Evaluating the Linear Regression model
print(linearModel.score(X_test, y_test))


# In[794]:


# Building and fitting the Ridge Regression model
ridgeModelChosen = Ridge(alpha = 2)
ridgeModelChosen.fit(X_train, y_train)

# Evaluating the Ridge Regression model
print(ridgeModelChosen.score(X_test, y_test))


# # Linear Regression

# In[795]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# # Ridge Regression 

# In[796]:



from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train, y_train)


# In[797]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# # Lasso Regression

# In[798]:



from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[961]:


X


# In[169]:


X.columns


# In[170]:


X.rename({'product type': 'PRODUCT_TYPE', 'Promotion applied': 'PROMOTION_APPLIED',"Generic Holiday":"GENERIC_HOLIDAY","DayOfWeek":"DAY_OF_WEEK"}, axis=1, inplace=True)


# In[171]:


X


# In[172]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# # Random Forest Regressor 

# In[173]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()


# # Hyper Parameter Tuning 

# In[174]:


from sklearn.model_selection import RandomizedSearchCV


# In[175]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(df_r) for df_r in np.linspace(start = 10, stop = 30, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(df_r) for df_r in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[176]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[177]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[178]:


rf_random.fit(X_train,y_train)


# In[179]:


rf_random.best_params_


# In[180]:


p=rf_random.predict(X_test)


# In[181]:


import sklearn.metrics as metrics


# In[182]:


metrics.r2_score(y_test,p)


# In[183]:


print('MAE:', metrics.mean_absolute_error(y_test,p))
print('MSE:', metrics.mean_squared_error(y_test,p))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,p)))


# In[184]:


zz=pd.DataFrame({'Actual':y_test,'Predicted':p})
zz.head(30)


# # Grdient Boosting Regressor 

# In[185]:


from sklearn.ensemble import GradientBoostingRegressor


# In[186]:


from sklearn.ensemble import GradientBoostingRegressor


# In[187]:


params={"n_estimators":150,"max_depth":4,"learning_rate":1,"criterion":"mse"}
gradient_boosting_regressor_model=GradientBoostingRegressor(**params)


# In[188]:


gradient_boosting_regressor_model.fit(X_train,y_train)


# In[189]:


PRED=gradient_boosting_regressor_model.predict(X_test)


# In[190]:


metrics.r2_score(y_test,PRED)


# In[191]:


print(gradient_boosting_regressor_model.score(X_train,y_train))


# In[192]:


print('MAE:', metrics.mean_absolute_error(y_test,PRED))
print('MSE:', metrics.mean_squared_error(y_test,PRED))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,PRED)))


# In[193]:


qq=pd.DataFrame({'Actual':y_test,'Predicted':PRED})
qq.head(30)


# In[194]:


print(merged.iloc[[76435]])


# In[199]:


print(merged.iloc[[719065]])


# In[ ]:





# In[ ]:





# In[195]:


def predict(PRODUCT_TYPE,PROMOTION_APPLIED,GENERIC_HOLIDAY,DAY_OF_WEEK):
    
    x= np.zeros(len(X.columns))
    x[0] = PRODUCT_TYPE
    x[1] = PROMOTION_APPLIED
    x[2] = GENERIC_HOLIDAY
    x[3] = DAY_OF_WEEK
   
    return gradient_boosting_regressor_model.predict([x])[0]


# In[196]:


predict(84,0,0,1)


# In[197]:


predict(120,1,0,4)


# In[198]:


predict(100,0,0,4)


# In[200]:


predict(790,0,0,2)


# In[201]:


import pickle
with open('BTACH_PICLKE','wb') as f:
    pickle.dump(gradient_boosting_regressor_model,f)


# In[ ]:




