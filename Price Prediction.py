# Dragon Real Estate - Price Predictor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

housing=pd.read_csv('dargon_real_estate_data.csv')
# print(housing.head(5))
# print(housing.info())
# print(housing['CHAS'].value_counts())
# print(housing.describe())

# housing.hist(bins=50,figsize=(20,15))
# plt.show()

# Train - Test Splitting

# *********** For learning purpose ******************
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled=np.random.permutation(len(data))
#     # print(shuffled)
#     test_set_size=int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     # print(test_indices)
#     train_indices=shuffled[test_set_size:]
#     # print(train_indices)
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set,test_set=split_train_test(housing,0.2)
# print('Rows in train set:',len(train_set))
# print('Rows in test set:',len(test_set))

#train_test_split randomly split the data which may cause the data to lose some important value
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42) #this gives two values
print('Rows in train set:',len(train_set))
print('Rows in test set:',len(test_set))

"""Random state is used to generate reproducable results and it can be any value"""

#Stratified shuffle split keeps the ratio of data maintained while splitting
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

# print('***Value count of Test set***\n',strat_test_set['CHAS'].value_counts())
# print('***Value count of Train set***\n',strat_train_set['CHAS'].value_counts())

housing=strat_train_set.copy()

# Looking for correlations
corr_matrix=housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes=['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()
housing.plot(kind="scatter",x='RM',y='MEDV',alpha=0.5)
plt.show()

# Trying out attribute combinations
housing['TAXRM']=housing['TAX']/housing['RM']
# print(housing.head(5))
corr_matrix=housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))
housing.plot(kind="scatter",x='TAXRM',y='MEDV',alpha=0.5)
plt.show()

housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()

# ****************** Dealing With Missing attributes ************************
""" To take care of missing attributes, we have three options:
    1. Get rid of the missing data points
    2. Get rid of the whole attribute
    3. Set the value to some value(0, mean or median)"""

# Option 1
# a=housing.dropna(subset=['RM'])
# print(a)
# # Note that the original housing dataframe will remain unchanged
#
# # Option 2
# housing.drop('RM',axis=1).shape
# # Note that the original housing dataframe will remain unchanged
#
# # # Option 3
# median=housing['RM'].median()
# housing['RM'].fillna(median)
# Note that the original housing dataframe will remain unchanged


#***************************************** Important *******************************************
# print(housing.describe()) #before started filling missing attributes
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)

""" The imputer class automatically replace empty value with mean
    in the entire dataset."""

print(imputer.statistics_)
X=imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)
print(housing_tr.describe())

# *********************** Scikit-Learn Design **************************

""" Primarily, three types of object in sklearn
    1. Estimators - It estimates some parameters based on dataset. Example: Imputer
        It has fit & tranform methods. 
        Fit method - fits the dataset and calculates internal parameters.
        
    2. Transformers - Transform method takes input and gives output based on the learnings from fit().
        It also has a convenience function called fit_transform() which fits and then transforms.
        
    3. Predictors - LinearRegression model is an example of predictors. Fit(), predict() are two common functions.
        It also gives score() function which will evaluate the predictions.
    """

# *********************** Feature scaling ***************************
""" Primarily, there are two types of feature scaling methods 
    1. Min-Max scaling (Normalization)
        (value-min)/(max-min)
        Gives value between 0 and 1
        for this sklearn provide a class called MinMaxScaler
    2. Standardizatio
        (value-mean)/std
        for this sklearn provide a class called StandardScalar
        """

# ***************************************** Creating pipeline ******************************************
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    # ...... add as many as you want in your pipeline
    ('std_scaler',StandardScaler()),
])

housing_num_tr=my_pipeline.fit_transform(housing)

# ***************************************** Selecting a desired model ******************************************
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
pred=model.predict(prepared_data)
print('predicted values are',list(pred),'\nand actual labels are',list(some_labels))

# ***************************************** Evaluating model ******************************************
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rsme=np.sqrt(mse)
# print(mse)

# Linear regression model is giving to much error while decision tree model overfits the data but giving better result
# Using better evaluation technique - Cross validation

from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
#Cv is number of divied parts or folds.
rsme_scores=np.sqrt(-scores)
# print(rsme_scores)

def print_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard Deviation:',scores.std())

print_scores(rsme_scores)

# Saving the model
# from joblib import dump,load
# dump(model,'dragon.joblib')

# Testing the model on test data
X_test=strat_test_set.drop('MEDV',axis=1)
Y_test=strat_test_set['MEDV'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rsme=np.sqrt(final_mse)

print(final_predictions,list(Y_test))
print(final_rsme)
print(prepared_data[0])