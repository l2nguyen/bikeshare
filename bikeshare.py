#------------------------------------------------------------------#
# KAGGLE COMPEITION - BIKESHARE
#
# Note: Data downloaded from Kaggle. This will look at the train
# and try to forecast bikeshare demand
#------------------------------------------------------------------#
# Created by by: Lena Nguyen, March 5, 2015
#------------------------------------------------------------------#

#----------------#
# IMPORT MODULES #
#----------------#

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

# Set current working directory
os.chdir("/Users/Zelda/Data Science/bikeshare")

#-------------#
# IMPORT DATA #
#-------------#

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = train.append(test)  # Make a dataframe with both train/test data together

len(train)  # 10886 obs
len(test)  # 6493 obs
len(data)  # 17379 obs

# Quick look at train
train.columns.values
train.dtypes

train.head(10)
train.describe()

# Quick look at test
test.columns.values
test.dtypes

test.head(10)
test.describe()

#---------------------#
# FEATURE ENGINEERING #
#---------------------#

# Define function to eparate date time variable to separate variables
def ts_transform(df, ts_var):
    df[ts_var] = pd.to_datetime(df[ts_var], format='%Y-%m-%d %H:%M:%S')
    df.set_index(ts_var, inplace=True)
    # Year
    df['Year'] = df.index.year
    # Month
    df['Month'] = df.index.month
    # Day
    df['Day'] = df.index.day
    # Hour
    df['Hour'] = df.index.hour
    # Day of week
    df['Weekday'] = df.index.weekday
    # Reset index
    df.reset_index(inplace=True)

# Format date of train dataset
ts_transform(train,'datetime')
train.head()
# Format date of test dataset
ts_transform(test,'datetime')
test.head()

# Average weekly temperature
for num in [1,2,6]:
    train[str('atemp'+'_time_'+str(num))] = pd.stats.moments.rolling_mean(train['atemp'],window=num,min_periods=1)
    test[str('atemp'+'_time_'+str(num))] = pd.stats.moments.rolling_mean(test['atemp'],window=num,min_periods=1)

#--------------------#
# DATA VISUALIZATION #
#--------------------#

# NOTE: Kaggle has a much nicer visualization of this at
# https://www.kaggle.com/users/993/ben-hamner/bike-sharing-demand/bike-rentals-by-time#script-save-run
train.groupby('Hour').casual.mean().plot(kind='bar',color='b')
plt.xlabel('Hour of the day')
plt.ylabel('Average number of casual riders')
plt.title('Average number of casual riders by hour')
plt.show()

train.groupby('Hour').registered.mean().plot(kind='bar',color='r')
plt.xlabel('Hour of the day')
plt.ylabel('Average number of registered riders')
plt.title('Average number of registered riders by hour')
plt.show()

#-- Graph shows continuous price increase since 1993, a short slight dip around 2009,
#-- Gas prices have been going down since Fall 2014

#----------#
# MODELING #
#----------#

cols = list(test.columns.values)
remove = [0,1,2,11]  # feats that proved to not be significant
feats = [i for j, i in enumerate(cols) if j not in remove]

# Train trainsets
X_train = train[feats]
Yc_train = train['casual']
Yr_train = train['registered']

# Test trainsets
X_test = test[feats]

# RF Model for casual riders
rfc = ensemble.RandomForestRegressor(n_estimators=100)
rfc.fit(X_train,Yc_train)

# RF Model for registered riders
rfr = ensemble.RandomForestRegressor(n_estimators=100)
rfr.fit(X_train,Yr_train)

# Look at R^2 values
rfc.score(X_train,Yc_train)
# 0.989
rfr.score(X_train,Yr_train)
# 0.993

# Look at feature importance
f_select = pd.DataFrame(data=rfc.feature_importances_, index=feats, columns=['Casual'])
f_select['Registered'] = rfr.feature_importances_
f_select


def cvrmse(X,Y,model):  # Function to get rmse for cross validation
    scores = cross_val_score(model, X, Y, cv=5, scoring='mean_squared_error')
    return np.mean(np.sqrt(-scores))

# Look at RMSE for casual and registered rider models
cvrmse(X_train, Yc_train, rfc)  # casual riders
# 22.049
cvrmse(X_train, Yr_train, rfr)  # resgistered riders
# 42.43

# Parameter tuning
depth_range = range(1,20)
maxfeats_range = range(1,13)
param_grid = dict(max_depth=depth_range, max_features=maxfeats_range)
# Casual riders
grid_cas = GridSearchCV(rfc, param_grid, cv=10, scoring='mean_squared_error')
grid_cas.fit(X_train, Yc_train)
# Registered riders
grid_reg = GridSearchCV(rfr, param_grid, cv=10, scoring='mean_squared_error')
grid_reg.fit(X_train, Yr_train)


#-----------------#
# MAKE SUBMISSION #
#-----------------#

# Predict test values
test['casual'] = np.round(rfc.predict(X_test))  # Casual riders
test['registered'] = np.round(rfr.predict(X_test))  # Registered riders
test['count'] = test['casual'] + test['registered']

submission = test.to_csv('submission5.csv', columns=['datetime','count'], index=False)
