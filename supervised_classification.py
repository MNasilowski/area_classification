# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:59:46 2021

@author: nasil
"""

import numpy as np
import pandas as pd
import time

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def undersampling(df):
# TODO Make it reusable
    forests = df[df["forest"]==1]
    water = df[df["water"]==1]
    fields = df[df["fields"]==1]
    other = df[(df["fields"]!=1) & (df["forest"]!=1) & (df["water"]!=1)]
    min_len = min(forests.shape[0], water.shape[0], fields.shape[0], other.shape[0])
    forests = forests.sample(min_len)
    water = water.sample(min_len)
    fields = fields.sample(min_len)
    other = other.sample(min_len)
    return water.append([fields,forests, other])
    
    
start_time = time.time()
df = pd.read_csv('data.csv')
df = undersampling(df)
X = df.iloc[:,1:14].to_numpy()
Y = df.iloc[:,-3:].to_numpy()


#Trees
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
clf = ExtraTreesClassifier(n_estimators=10, max_depth=10, min_samples_leaf=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_true=y_test[:,2], y_pred=y_pred[:,2])

            
#Random Forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_true=y_test[:,0], y_pred=y_pred[:,0])
