
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import copy


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


rawdata = pd.read_csv('problem_dataset.csv')


# In[7]:


rawdata.head()


# In[8]:


rawdataew = copy.copy(rawdata)


# In[9]:


rawdataew = rawdataew.dropna()


# In[10]:


rawdataew.head()


# In[11]:


rawdata.set_index('Date')


# In[12]:


def format_str(x):
    x = str(x)
    if 'M' in x:
        return float(x[0:(-1)])*10**7
    elif 'K' in x:
        return float(x[0:(-1)])*10**3
    
def last_letter(x):
    x = str(x)
    return x[-1]


# In[18]:


feature_6_idx = df['feature_6'].notnull()
feature_6_notnull = df['feature_6'].loc[feature_6_idx]
np.unique(feature_6_notnull.apply(last_letter))


# In[15]:


format_str('1.94K')


# In[16]:


df = copy.copy(rawdata)


# In[17]:


df['feature_6'] = df['feature_6'].apply(format_str)


# In[19]:


df['feature_1'].plot.hist(grid=True,color='r', alpha=0.5, label='feature_1',bins=8)
plt.show()


# In[20]:


max(df['feature_1'].loc[df['feature_1'].notnull()])


# In[21]:


min(df['feature_1'].loc[df['feature_1'].notnull()])


# In[22]:


np.sort(df['feature_1'].loc[df['feature_1'].notnull()])[::-1][0:10]


# In[23]:


df['feature_1'].loc[df['feature_1']>float(10**4)] = np.nan


# In[24]:


df['feature_1'].plot.hist(grid=True,color='r', alpha=0.5, label='feature_1',bins=20)
plt.show()


# In[25]:


def kunit(x):
    x = str(x)
    if ',' in str(x):
        x = x.replace(',', '')
        return float(x)
    


# In[26]:


df['feature_2'] = df['feature_2'].apply(kunit)


# In[27]:


np.sort(df['feature_2'].loc[df['feature_2'].notnull()])[::-1]


# In[28]:


df['feature_2'].plot.hist(grid=True,color='r', label='feature_2', bins=20)


# In[29]:


fig2 = plt.figure(figsize=(100,30))
plt.plot(df['feature_2'], 'o-', color='red', label='feature_2')
plt.grid();
plt.legend();
plt.show()


# In[30]:


np.sort(df['feature_3'].loc[df['feature_3'].notnull()])[::-1][0:20]


# In[31]:


len(df['feature_2'].loc[df['feature_3']>150])


# In[32]:


df['feature_3'].plot.hist(grid=True,color='r', label='feature_3', bins=20)


# In[33]:


np.sort(df['feature_3'].loc[df['feature_3'].notnull()])[::-1]


# In[34]:


df['feature_4'].plot.hist(grid=True,color='r', label='feature_4', bins=20)


# In[35]:


np.sort(df['feature_5'].loc[df['feature_5'].notnull()])[::-1]


# In[36]:


df['feature_5'].loc[df['feature_5']<0] = 0.0


# In[37]:


df['feature_5'].plot.hist(grid=True,color='r', label='feature_5', bins=20)


# In[38]:


fig5 = plt.figure(figsize=(100, 30))
plt.plot(df['feature_5'], 'o-', color='red', label='feature_5')
plt.grid();
plt.legend();
plt.show()


# In[39]:


np.sort(df['feature_6'].loc[df['feature_6'].notnull()])[::-1]


# In[40]:


df['feature_6'].plot.hist(grid=True,color='r', label='feature_6', bins=20)


# In[41]:


fig6 = plt.figure(figsize=(100, 30))
plt.plot(df['feature_6'], 'o-', color='red', label='feature_6')
plt.grid();
plt.legend();
plt.show()


# In[42]:


np.sort(df['feature_7'].loc[df['feature_7'].notnull()])[::-1]


# In[43]:


df['feature_7'].plot.hist(grid=True, color='r', label='feature_7', bins=20)


# In[44]:


fig6 = plt.figure(figsize=(100, 30))
plt.plot(df['feature_7'], 'o-', color='red', label='feature_7')
plt.grid();
plt.legend();
plt.show()


# In[45]:


fig6 = plt.figure(figsize=(100, 30))
plt.plot(df['target'], 'o-', color='red', label='target')
plt.grid();
plt.legend();
plt.show()


# In[46]:


df.head()


# In[47]:


df.set_index('Date', inplace=True)


# In[48]:


df.index = pd.to_datetime(df.index).date
df.sort_index(inplace=True)


# In[49]:


df.head()


# In[50]:


idx = np.where((np.isnan(df['feature_8'])==False)&(df['feature_8']>=-1e7))[0]
xp = np.array(df['feature_8'].iloc[idx])
xvals = np.arange(len(df))
yvals = np.interp(xvals, idx, xp)


# In[51]:


fig5 = plt.figure(figsize=(100,30))
plt.plot(df.index, yvals, 'o-', color='red', label='feature_8')
plt.grid();
plt.legend();
plt.show()


# In[52]:


df['feature_8'] = yvals


# In[53]:


idx = np.where((np.isnan(df['feature_9'])==False)&(df['feature_9']>=-1e7))[0]
xp = np.array(df['feature_9'].iloc[idx])
xvals = np.arange(len(df))
yvals = np.interp(xvals, idx, xp)


# In[54]:


fig9 = plt.figure(figsize=(100,30))
plt.plot(df.index, yvals, 'o-', color='red', label='feature_9')
plt.grid();
plt.legend();
plt.show()


# In[55]:


df['feature_9'] = yvals


# In[56]:


df.head()


# In[57]:


df = df.fillna(method='ffill')


# In[58]:


df.head()


# In[59]:


df.corr()


# In[60]:


df.columns


# In[61]:


df = df.drop(df.index[0])


# In[62]:


np.array(df['feature_1'].quantile(q=[0.01, 0.99]))


# In[58]:


def quantile_clipping(s):
    qs = np.array(s.quantile(q=[0.01, 0.99]))
    s.loc[s<qs[0]] = qs[0]
    s.loc[s>qs[1]] = qs[1]
    return s


# In[59]:


df.apply(quantile_clipping, axis=1)


# In[63]:


from sklearn.linear_model import RidgeCV, LinearRegression, Ridge


# In[64]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[70]:


alphas = np.logspace(-10, -2, 200)


# In[66]:


def Ridge_CV(X, y, alphas, percent):
    X = np.array(X)
    y = np.array(y)
    assert X.shape[0] == len(y)
    n = len(y)
    m = len(alphas)
    mse = np.zeros(m)
    mae = np.zeros(m)
    rmse = np.zeros(m)
    n_train = int(percent*n)
    X_train = X[0:n_train, :]
    y_train = y[0:n_train]
    X_valid = X[n_train:,:]
    y_valid = y[n_train:]
    for i in range(m):
        model = Ridge(alpha=alphas[i])
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        mse[i] = mean_squared_error(pred, y_valid)
        mae[i] = mean_absolute_error(pred, y_valid)
        rmse[i] = np.sqrt(mse[i])
    return mse, mae, rmse


# In[67]:


mse, mae, rmse = Ridge_CV(df.iloc[:, 0:7], df['target'], alphas, 0.7)


# In[68]:


rmse


# In[69]:


np.argmin(rmse)


# In[72]:


X = np.array(df.iloc[:, 0:7])
y=np.array(df['target'])
n = len(y)
m = len(alphas)
n_train = int(0.7*n)
X_train = X[0:n_train, :]
y_train = y[0:n_train]
X_valid = X[n_train:,:]
y_valid = y[n_train:]


# In[73]:


def Ridge_single(alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mse = mean_squared_error(pred, y_valid)
    mae = mean_absolute_error(pred, y_valid)
    rmse = np.sqrt(mse)
    return [mse, mae, rmse]


# In[74]:


import multiprocessing as mp


# In[75]:


p = mp.Pool(30)
ans = np.array(p.map(Ridge_single, alphas))


# In[80]:


rmses = ans[:,2]
mae = ans[:,1]


# In[81]:


rmses


# In[83]:


model = Ridge(alpha=alphas[0])


# In[85]:


model.fit(X_train, y_train)
t_pred = model.predict(X)


# In[86]:


model.get_params()


# In[87]:


fig_tpred = plt.figure(figsize=(100,30))
plt.plot(df.index, t_pred, 'o-', color='r', label='prediction')
plt.plot(df.index, df['target'], 'o-', color='blue', label='actual')
plt.grid();
plt.legend();
plt.show()


# In[135]:


f7_pred = model.predict(np.array(df.iloc[:,0:7]))


# In[136]:


fig_f7pred = plt.figure(figsize=(100,30))
plt.plot(df.index, f7_pred, 'o-', color='r', label='prediction')
plt.plot(df.index, df['target'], 'o-', color='blue', label='actual')
plt.grid();
plt.legend();
plt.show()


# In[191]:


X = np.array(df.iloc[:, 1:6])
y=np.array(df['target'])
n_train = int(percent*n)
X_train = X[0:n_train, :]
y_train = y[0:n_train]
X_valid = X[n_train:,:]
y_valid = y[n_train:]


# In[192]:


def Ridge_single(alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(pred, y_valid)
    mae = mean_absolute_error(pred, y_valid)
    rmse = np.sqrt(mse[i])
    return [mse, mae, rmse]


# In[2]:


import multiprocessing as mp


# In[3]:


p = mp.Pool(30)
ans = np.array(p.map(Ridge_single, alphas))
mse = ans[:,0]
mae = ans[:,1]
rmse = ans[:,2]


# In[119]:


np.array(df['target'].rolling(30).mean())


# In[82]:


def Bollinger(series, n_window, n_std):
    n = len(series)
    
    roll_mean = series.rolling(n_window).mean()
    roll_std = series.rolling(n_window).std()
    upper = np.array(roll_mean+n_std*roll_std)
    lower = np.array(roll_mean-n_std*roll_std)
    prev = 0
    positions = np.zeros(n)
    s = np.array(series)
    prev = 0
    prevs = np.zeros(n)
    for i in range(1, (n-1)):
        if (s[i]>upper[i]) and (s[(i-1)]<=upper[(i-1)]):
            if prev == 0:
                positions[(i+1)] = -1
                prevs[(i+1)] = -1
                prev = -1
            elif prev == 1:
                positions[(i+1)] = -2
                prevs[(i+1)] = -1
                prev = -1
        elif (s[(i-1)]>upper[(i-1)]) and (s[i]<=upper[i]):
            if prev == -1:
                positions[(i+1)] = 1
                prevs[(i+1)] = 0
                prev = 0
        elif  (s[i]<lower[i]) and (s[(i-1)]>=lower[(i-1)]):
            if prev == 0:
                positions[(i+1)] = 1
                prevs[(i+1)] = 1
                prev = 1
            elif prev == -1:
                positions[(i+1)] = 2
                prevs[(i+1)] = 1
                prev = 1
        elif (s[(i-1)]<lower[(i-1)]) and (s[i]>=lower[i]):
            if prev == 1:
                positions[(i+1)] = -1
                prevs[(i+1)] = 0
                prev = 0
        else:
            prevs[(i+1)] = prevs[i]
    return {'positions':positions, 'upper':upper, 'lower':lower, 'pos':prevs}
            


# In[84]:


trade['positions'][0:200]


# In[85]:


trade['pos'][0:200]


# In[94]:


trade = Bollinger(df['target'], 30, 1.8)


# In[95]:


pnl=-np.cumsum(df['target']*trade['positions'])+df['target']*trade['pos']


# In[96]:


fig = plt.figure()
plt.plot(pnl)


# In[97]:


def momentum_sig(series, long, short):
    s = np.array(series)
    n = len(s)
    mean_long = np.array(series.rolling(long).mean())
    mean_short = np.array(series.rolling(short).mean())
    prev = 0 
    signals = np.zeros(n)
    for i in range(1, (n-1)):
        if (mean_short[i] > mean_long[i]) and (mean_short[(i-1)]<= mean_long[(i-1)]):
            if prev == 0:
                signals[(i+1)] = 1
                prev = 1
        elif (mean_short[i] <= mean_long[i]) and (mean_short[(i-1)] > mean_long[(i-1)]):
            if prev == 1:
                signals[(i+1)] = -1
                prev = 0
    return {'signals':signals, 'mean_long':mean_long, 'mean_short':mean_short}


# In[99]:


momentum_trade = momentum_sig(df['target'], 15, 5)


# In[101]:


momentum_trade['signals'][0:100]


# In[109]:


def long_only_pnl(signals, prices, init, tcost):
    n = len(signals)
    cash = np.zeros(n)
    cash[0] = init
    portfolio = np.zeros(n)
    for i in range(1,n):
        if signals[i] == 1:
            cash[i] = 0
            portfolio[i] = cash[(i-1)]/(prices[i]*(1+tcost))
        elif signals[i] == -1:
            cash[i] = portfolio[(i-1)]*prices[i]/(1-tcost)
            portfolio[i] = 0
        else:
            portfolio[i] = portfolio[(i-1)]
            cash[i] = cash[(i-1)]
    return cash+portfolio*prices


# In[112]:


pnl = long_only_pnl(momentum_trade['signals'], df['target'], 100,  0.00)


# In[113]:


fig_pnl = plt.figure(figsize=(100,30))
plt.plot(df.index, pnl, '-', label='pnl', color='red')

plt.grid();
plt.legend();
plt.show()


# In[261]:


fig_pnl = plt.figure(figsize=(100,30))
plt.plot(df.index, df['target'], '-', label='pnl', color='red')
plt.plot(df.index, trade['upper'], '-', label='pnl', color='blue')
plt.plot(df.index, trade['lower'], '-', label='pnl', color='blue')

plt.grid();
plt.legend();
plt.show()


# In[262]:


fig_pnl = plt.figure(figsize=(100,30))
plt.plot(df.index, trade['positions'], 'o-', label='pnl', color='red')
plt.grid();
plt.legend();
plt.show()


# In[258]:


min(np.cumsum(trade['positions']))


# In[259]:


trade['positions'][0:100]

