
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


# In[22]:


rawdata = pd.read_csv('problem_dataset.csv')


# In[23]:


def format_str(x):
    x = str(x)
    if 'M' in x:
        return float(x[0:(-1)])*10**7
    elif 'K' in x:
        return float(x[0:(-1)])*10**3
    
def last_letter(x):
    x = str(x)
    return x[-1]


# In[36]:



df = copy.copy(rawdata)
df.head()
df = df.drop(columns=['feature_2'])


# In[37]:


df['feature_6'] = df['feature_6'].apply(format_str)


# In[38]:


df = df.dropna(axis=0)


# In[35]:


df


# In[39]:


X = np.array(df.iloc[:,1:7])


# In[40]:


y = np.array(df['target'])


# In[41]:


alphas = np.logspace(-7, -2, 100)


# In[42]:


from sklearn.linear_model import RidgeCV


# In[43]:


model = RidgeCV(normalize=True, alphas=alphas)


# In[44]:


model.fit(X,y)


# In[45]:


X.shape[0]


# In[46]:


model.coef_


# In[47]:


model.alpha_


# In[48]:


alphas


# In[49]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[50]:


param_grid = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [2, 5, 7, 9]
}


# In[52]:


from sklearn.model_selection import GridSearchCV
clf = RandomForestRegressor()
grid_clf = GridSearchCV(clf, param_grid, cv=10)
grid_clf.fit(X,y)


# In[55]:


grid_clf.best_params_


# In[59]:


model = RandomForestRegressor(n_estimators=100)

cv = model_selection.KFold(n_splits=3)
predictors = np.array(np.arange(len(y)))
for train_index, test_index in cv.split(predictors):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    # For training, fit() is used
    model.fit(X_train, y_train)

    # Default metric is R2 for regression, which can be accessed by score()
    model.score(X_test, y_test)
 
    # For other metrics, we need the predictions of the model
    y_pred = model.predict(X_test)

    print(metrics.mean_squared_error(y_test, y_pred))
    print(metrics.r2_score(y_test, y_pred))


# In[60]:


dummy_df = pd.DataFrame({'cat':['a','b', 'a'], 'num':[1,2,3]})


# In[61]:


dummy_df


# In[63]:


dummy_df = dummy_df.join(pd.get_dummies(dummy_df['cat']))


# In[64]:


dummy_df


# In[65]:


dummy_df = dummy_df.drop(columns=['cat'])


# In[66]:


dummy_df


# In[67]:


dummy_df = dummy_df.drop([0])


# In[69]:


dummy_df= dummy_df.drop(df.index==1)

