import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""modelname = smf.ols(formula='QUANT_RESPONSE ~ C(CAT_EXPLANATORY)', data = dataframe.fit())"""
#read csv and set variable 'data' to the read file
data = pd.read_csv('dengue_features_train_1.csv', low_memory = 'False', sep = ',')
data1 = pd.read_csv('dengue_features_test.csv', low_memory = 'False', sep = ',')
# change data index to numeric instead of strings 
# coerce basically forces the blanks to be NaN
data.apply(lambda x: pd.to_numeric(x, errors='ignore'))
data1.apply(lambda x: pd.to_numeric(x, errors='ignore'))
data_clean = data.dropna()
data1_clean = data1filter.dropna()
# centre the variables to make them comparable
def centrevar(x):
    # y = x + '_c'
    data_clean[x] = (data_clean[x] - data_clean[x].mean())
   
x = list(data_clean)
y = []
for i in x[4:-2]:
    centrevar(i)
    # to_list = i + '_c'
    # y.append(to_list)
z = list(data1_clean)
for i in z[4:]:
    centrevar(i)    
# regstr = 'total_cases ~ ' + ' + '.join(y[:-1])

# reg2 = smf.ols(formula = regstr, data = data_clean).fit()
# print (list(data_clean))

# print (reg2.summary())

datasub = data_clean[data_clean.columns[4:-2]]
data1sub = data1_clean[data1_clean.columns[4:-1]]
# print (list(datasub))
# ML part
trainingData = np.array(datasub)
trainingScores = np.array(data_clean['total_cases'])
print (trainingScores)
clf = LinearRegression(fit_intercept=True)
clf.fit(trainingData,trainingScores)

predictionData = np.array(data1sub)
print (clf.predict(predictionData))