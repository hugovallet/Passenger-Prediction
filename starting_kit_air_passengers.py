
# coding: utf-8

# # <a href="http://www.datascience-paris-saclay.fr">Paris Saclay Center for Data Science</a>
# # <a href=http://www.datascience-paris-saclay.fr/en/site/newsView/12>RAMP</a> on predicting the number of air passengers
# 
# <i> Balázs Kégl (LAL/CNRS), Alex Gramfort (LTCI/Telecom ParisTech), Djalel Benbouzid (UPMC), Mehdi Cherti (LAL/CNRS) </i>

# ## Introduction
# The data set was donated to us by an unnamed company handling flight ticket reservations. The data is thin, it contains
# * the date of departure
# * the departure airport
# * the arrival airport
# * the mean and standard deviation of the number of weeks of the reservations made before the departure date
# * a field called <code>log_PAX</code> which is related to the number of passengers (the actual number were changed for privacy reasons)
# 
# The goal is to predict the <code>log_PAX</code> column. The prediction quality is measured by RMSE. 
# 
# The data is obviously limited, but since data and location informations are available, it can be joined to external data sets. **The challenge in this RAMP is to find good data that can be correlated to flight traffic**.

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)






# In[3]:

# optional
import seaborn as sns; sns.set()


# ## Fetch the data and load it in pandas

#%% start !

data = pd.read_csv("./public_train.csv",index_col=0)

data = pd.read_csv("./public_train.csv")

print min(data['DateOfDeparture'])
print max(data['DateOfDeparture'])

data.head()
data['Departure'].unique()
data.hist(column='log_PAX', bins=50);
data.hist('std_wtd', bins=50);
data.hist('WeeksToDeparture', bins=50);
data.describe()
data.dtypes
data.shape

print data['log_PAX'].mean()
print data['log_PAX'].std()


# ## Preprocessing for prediction

# Getting dates into numerical columns is a common operation when time series are analyzed with non-parametric predictors. The code below makes all possible choices: ordered columns for the year, month, day, weekday, week, and day in the year, and one-hot columns for year month, day, weekday, and week.
# 
# The departure and arrival airports are also converted into one-hot columns. 

# In[54]:

data_encoded = data

data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
data_encoded = data_encoded.drop('Departure', axis=1)
data_encoded = data_encoded.drop('Arrival', axis=1)

# following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))


# In[55]:

data_encoded.tail(5)


# ### A linear regressor baseline
# 
# We drop the target column and the original data column.

# In[56]:

features = data_encoded.drop(['log_PAX','DateOfDeparture'], axis=1)
X_columns = data_encoded.columns.drop(['log_PAX','DateOfDeparture'])
X = features.values
y = data_encoded['log_PAX'].values





#%%
from bokeh.charts import *
from bokeh.plotting import *

date = pd.to_datetime(data_encoded["DateOfDeparture"], '%Y-%m-%d')
date.sort()
date.index = np.arange(0,11128)

output_file("to predict.html")



data_plot = dict(AAPL=data_encoded['log_PAX'], Date=date)

p = Line(data_plot)

show(p)
#%%
f,axis=plt.subplots()
daxis=data_encoded.boxplot(column="log_PAX",by="DateOfDeparture",ax=axis)
plt.show(axis)

# It gives us a pretty nice imporvement above baseline


#%%
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

reg = LinearRegression()

scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error')
print("log RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))


# Exercise: Visualize the coefficients, try to make sense of them.

# ## Random Forests





from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(n_estimators = 100,max_depth =100, max_features ="auto")

scores = cross_val_score(reg, X_train, y_train, cv=3, scoring='mean_squared_error',n_jobs=1)
print("Random forest RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))

# ## Variable importances

#%%

reg.fit(X_train, y_train)


len(X_columns)



plt.figure(figsize=(15, 5))

ordering = np.argsort(reg.feature_importances_)[::-1][:50]

importances = reg.feature_importances_[ordering]
feature_names = X_columns[ordering]

x = np.arange(len(feature_names))
plt.bar(x, importances)
plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);

prediction=reg.predict(X_train)

# # Submission
# 
# ### The feature extractor
# 
# The feature extractor implements a single <code>transform</code> function. It receives the full pandas object X_df (without the labels). It should produce a numpy array representing the features extracted. If you want to use the (training) labels to save some state of the feature extractor, you can do it in the fit function. 
# 
# 
# You can choose one of the example feature extractors and copy-paste it into your feature_extractor.py file.

#%% show the prediction's quality 

output_file("prediction.html")
p = figure(plot_width=1600, plot_height=800)
p.line(list(np.arange(0,len(y_train))),list(y_train),line_color="blue",alpha=0.2,name="True values",visible=True)
p.line(list(np.arange(0,len(y_train))),list(prediction),line_color="red",alpha=0.2,name="Predicted values")
#p.legend()


show(p)


# In[63]:

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array


# This feature extractor shows you how to join your data to external data. You will have the possibility to submit a single external csv for each of your submission (so if you have several data sets, you first have to do the join offline, and save is as a csv). In this case it is whether data, joined to the database on the <code>DateOfDeparture</code> and <code>Arrival</code> fields. Attention: when you join the data, make sure that the <b><font color=red>order</font> of the rows in the data frame does not change</b>.

# In[64]:

import pandas as pd
import os


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        # path = os.path.dirname(__file__)  # use this in submission
        path = '.'  # use this in notebook
        data_weather = pd.read_csv(os.path.join(path, "external_data.csv"))
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
        
        X_weather = X_weather.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        X_encoded = X_encoded.set_index(['DateOfDeparture', 'Arrival'])
        X_weather = X_weather.set_index(['DateOfDeparture', 'Arrival'])
        X_encoded = X_encoded.join(X_weather).reset_index()

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)

        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values
        return X_array


# ### The regressor
# 
# The regressor should implement an sklearn-like regressor with fit and predict functions. You can copy paste either of these into your first regressor.py file.

# In[65]:

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestRegressor(n_estimators=100, max_depth=100, max_features=100)
        self.clf = AdaBoostRegressor(n_estimator=100,loss="exponential")

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


# In[66]:

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = LinearRegression()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)


# Let's put it together and run the chain.

# In[67]:

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit


def train_model(X_df, y_array, skf_is):
    fe = FeatureExtractor()
    fe.fit(X_df, y_array)
    X_array = fe.transform(X_df)
    # Regression
    train_is, _ = skf_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    reg = Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg


def test_model(trained_model, X_df, skf_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df)
    # Regression
    _, test_is = skf_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array


data = pd.read_csv("public_train.csv")
X_df = data.drop(['log_PAX'], axis=1)
y_array = data['log_PAX'].values

skf = ShuffleSplit(y_array.shape[0], n_iter=2, test_size=0.2)
skf_is = list(skf)[0]

trained_model = train_model(X_df, y_array, skf_is)
y_pred_array = test_model(trained_model, X_df, skf_is)
_, test_is = skf_is
ground_truth_array = y_array[test_is]

score = np.sqrt(np.mean(np.square(ground_truth_array - y_pred_array)))
print 'RMSE =', score


# ## Unit testing
# 
# It is <b><span style="color:red">important that you test your submission files before submitting them</span></b>. For this we provide a unit test. Place the python files <code>regressor.py</code>, <code>feature_extractor.py</code>, <code>external_data.csv</code> and <a href="https://drive.google.com/file/d/0BzwKr6zuOkdRdUN2WEtuZlVHUmM/view?usp=sharing"><code>user_test_model.py</code></a> in a directory, set the paths to the data files in <code>user_test_model.py</code>, and run 
# 
# <code>python user_test_model.py</code>
# 
# If it runs and prints 
# <code>
# RMSE = [some_number_hopefully_below_1]
# </code>
# you can submit the code.

