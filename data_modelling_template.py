# -*- coding: utf-8 -*-
"""
data_modelling_template

"""


#==============================================================================
# Statsmodels

# issues-
# use the api or the r/patsy approach (autocreation of dummys)
#==============================================================================

#groupwise linear regression

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params
close_px = pd.read_csv('pydata-book-master/ch09/stock_px.csv', parse_dates=True, index_col=0)    
by_year = close_px.groupby(lambda x: x.year)
by_year.apply(regress, 'AAPL',['SPX'])

#select xvar list
X = close_px[['SPX','XOM']]
X['intercept'] = 1.
Y = close_px['AAPL']
result = sm.OLS(Y, X).fit()

result.summary()

#==============================================================================
# Strategies for feature building
#==============================================================================
patsy v normal api
import statsmodels.formula.api as smf
y, X = dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
smf.OLS(y,X)

# create multiple dummy variables in the normal api style
for column in ['Name', 'Year']:
    ...:     dummies = pd.get_dummies(df[column])
    ...:     df[dummies.columns] = dummies
    
    
#==============================================================================
# logit regression
#==============================================================================
# assume 'admit' already 1, 0
data['intercept'] = 1.0
logit = sm.Logit(data['admit'], data[train_cols])
# odds ratios only
print np.exp(result.params)

#regularization
#The regularization parameter alpha should be a scalar or have the same shape as results.params
alpha = 0.1 * len(spector_data.endog) * np.ones(spector_data.exog.shape[1])
#Choose not to regularize the constant
alpha[-1] = 0
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_l1_res = logit_mod.fit_regularized(method='l1', alpha=alpha)





#==============================================================================
#Sckit learn 
#==============================================================================
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_features="auto",
                                  min_samples_leaf=10)
clf.fit(df[iris.feature_names], df.species)

from sklearn.externals.six import StringIO
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
    
    
from sklearn.datasets import load_boston
boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names[:13])
# add in prices
df['price'] = boston.target

from sklearn.linear_model import LinearRegression

features = ['AGE', 'LSTAT', 'TAX']
lm = LinearRegression()
lm.fit(df[features], df.price)

# add your actual vs. predicted points
pl.scatter(df.price, lm.predict(df[features]))
# add the line of perfect fit
import matplotlib.pyplot as pl
straight_line = np.arange(0, 60)
pl.plot(straight_line, straight_line)
pl.title("Fitted Values")

#or use ggplot!!
df['predict'] = lm.predict(df[features])
p = ggplot(df, aes(df.price, df.predict )) + geom_point(color='black') + stat_smooth(color='red')


from sklearn import linear_model
clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit(df[features], df.price)
#or 
clf = linear_model.Lasso(alpha = 0.1)

