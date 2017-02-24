#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:58:31 2017

@author: rama
"""

import os
import pandas as pd
import requests
import numpy as np
PATH = r'/home/rama/git/mycode/ds_code/data/'
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)

os.chdir(PATH)
df = pd.read_csv(PATH + 'iris.data', names=['sepal length', 'sepal width',
'petal length', 'petal width', 'class'])
df.head()
df.head().T



import seaborn as sns
sns.pairplot(df,hue='class')

df['class'] = df['class'].map({'Iris-setosa':'SET', 'Iris-versicolor':'VIR', 'Iris-virginica':'VER'})

df.groupby('class').mean()

df.groupby('class')['petal width']\
.agg({'delta': lambda x: x.max() - x.min(), 'max': np.max, 'min': np.min})