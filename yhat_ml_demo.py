import pandas as pd
pd.options.display.max_columns = 50
pd.options.display.width = 500 
# or pd.set_options('max_columns',50)ls
df = pd.read_csv("/Users/stevegoodman/Documents/Dev/kaggle/cs-training.csv")
df.head()
df.SeriousDlqin2yrs.mean()
pd.crosstab(df.NumberOfTimes90DaysLate, df.SeriousDlqin2yrs)
pd.value_counts(df.NumberOfDependents).plot(kind='bar')
#top 3 by count
pd.value_counts(df.NumberOfDependents).head(3)

#sci-kit learn demos
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

svm_clf = SVC()
neighbors_clf = KNeighborsClassifier()
clfs = [
    ("svc", SVC()),
    ("KNN", KNeighborsClassifier())
    ]
for name, clf in clfs:
    clf.fit(df[iris.feature_names], df.species)
    print name, clf.predict(iris.data)
    print "*"*80


#linear regression
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)



from sklearn.linear_model import LinearRegression
import re


def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

df = pd.DataFrame(boston.data)
df.columns = [camel_to_snake(col) for col in boston.feature_names[:-1]]
# add in prices
df['price'] = boston.target
features = ['age', 'lstat', 'tax']
lm = LinearRegression()
lm.fit(df[features], df.price)
import pylab as pl

#hmm titling not working correctly
pl.title("Fitted Values")
pl.scatter(df.price, lm.predict(df[features]))