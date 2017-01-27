import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import interactive
interactive(True)


startup_data = pd.read_csv('50_Startups.csv')

sns.pairplot(startup_data, hue='State', diag_kind='hist')

X = startup_data.iloc[:, :-1].values
y = startup_data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the dummy variable trap
X = X[:,1:] 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_text = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(np.ones((X.shape[0],1)), values=X, axis=1)


# X_opt = X[:, [0,1,2,3,4,5]]
# regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
# regressor_OLS.pvalues
 
 def backwardElimination_rama(x, sl):
     condition = True
     while condition:
        regressor_OLS = sm.OLS(y, x).fit()
        maxP = max(regressor_OLS.pvalues).astype(float)
        maxp_inx = np.argmax(regressor_OLS.pvalues, axis=0)
        if maxP > sl and x.shape[1] > 0:
            x = np.delete(x, maxp_inx, axis=1)
        else:
            condition = False
     return x            

 X = backwardElimination_rama(X, 0.05)
print(X)            
# regressor_OLS.summary()
# X_opt = X[:, [0,1,3,4,5]]
# regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
# regressor_OLS.summary()
# X_opt = X[:, [0,3,4,5]]
# regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
# regressor_OLS.summary()
# X_opt = X[:, [0,3,5]]
# regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
# regressor_OLS.summary()
# X_opt = X[:, [0,3]]
# regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
# regressor_OLS.summary()


#Automated Backward Elimination

#==============================================================================
#  def backwardElimination(x, sl):
#      numVars = len(x[0])
#      for i in range(0, numVars):
#          regressor_OLS = sm.OLS(y, x).fit()
#          maxVar = max(regressor_OLS.pvalues).astype(float)
#          if maxVar > sl:
#              for j in range(0, numVars - 1):
#                  if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                      x = np.delete(x, j, 1)
#      regressor_OLS.summary()
#      return x
#   
#  SL = 0.05
#  X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#  X_Modeled = backwardElimination(X_opt, SL)
# 
#==============================================================================
