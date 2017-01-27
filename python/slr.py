import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import seaborn as sns

salary_data = pd.read_csv('Salary_Data.csv')

#salary_data.plot(x='YearsExperience',y='Salary',kind='scatter')
#sns.palplot(sns.diverging_palette(2,100))
#sns.boxplot(data=salary_data)
#sns.distplot(salary_data.iloc[:,0], kde=True, rug=True)
#sns.jointplot(x='YearsExperience', y='Salary', data=salary_data)


X = salary_data.iloc[:,:-1]
y = salary_data.iloc[:, 1]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
