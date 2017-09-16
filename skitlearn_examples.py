# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 19:09:09 2014

@author: stevegoodman
"""

import numpy

from sklearn import datasets
diabetes = datasets.load_diabetes()
data = diabetes.data
data.shape
diabetes_X = diabetes.data
diabetes_y = diabetes.target


diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

from sklearn.cross_validation import train_test_split
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X, diabetes_y)


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
regr.score(diabetes_X_test, diabetes_y_test)


 alphas = np.logspace(-4, -1, 6)
from __future__ import print_function
print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]) 
