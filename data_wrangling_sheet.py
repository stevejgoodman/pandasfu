# -*- coding: utf-8 -*-
"""
Common wrangling scenarios
http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_04_wrangling.ipynb
http://nbviewer.ipython.org/urls/gist.github.com/fonnesbeck/5850413/raw/3a9406c73365480bc58d5e75bc80f7962243ba17/2.+Data+Wrangling+with+Pandas.ipynb
"""

"""Areas still to cover:
Output to Excel libs
Dedupe lib
Seaborn/ggplot
Scipy, statsmodels
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import sklearn as skl
import seaborn as sns
#required for notebook usage
ipython notebook --pylab=inline
%matplotlib inline

#tell pandas to display wide tables as pretty HTML tables
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

stack/pivot
groupby (), custom agg function for group by, first/last in group
group by summary/detail combined


#REM - don't forget numpy & scipi functions, and statsmodels and excel output


#------------------------------------------
#1. Load/Save data from csv/database/mongo

pieces=[]
columns=['name','sex','birth']
for year in years:
    os.path.abspath(os.path.join('Users','stevegoodman','Documents','Dev','pydata-book-master','ch02','names','yob%d.txt') % year)    
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)



#convertors
chicago = pd.read_csv('city-of-chicago-salaries.csv',
                      header=False,
                      names=headers,
                      converters={'salary': lambda x: float(x.replace('$', ''))})


#type casting a series
data['quarter'] = data['quarter'].astype('int64')
#Ideally load the data types properly in the first place - but this appears to be a bug....
data = pd.read_csv('pydata-book-master/ch08/macrodata.csv', dtype={'year':np.int64, 'quarter':np.int64})
#Workarounds
data['year']= data['year'].astype('int64')
df['a'].convert_objects(convert_numeric='force')


engine = create_engine("postgresql://postgres@localhost/Test")

#careful, non numerics have been assigned as type object - obviously not what I want 
df = pd.read_sql_table('users', engine)


#---------------------------------------------------------------------------
#2. Column transformation and Mapping, drop/infill missings, indexes

#Combine cut with get_dummies see Wes p 205
#Not sure how to separate out np.nans - bins doesnt like non-numerics
values = np.random.rand(10)
values[1]=np.nan

bins = [0,0.2,0.6,1]
pd.get_dummies(pd.cut(values,bins))




#column filtering
movies[ ['release_date','title'] ]
#renaming
movies.rename(columns={'video_release_date':'vid_rd'}, inplace=True)

movies.sort_index(by='title').head()
#select rows that contain values a,b,c
mask = col.isin(['a','b','c']); df[mask]

# Missing or bad data ----------------------

#default is to drop any rows with missings, but can specify all variables must be missing
data.dropna(how='all')
#or drop a column if all empty, thres=3 will only keep series with at least 3 non missing vals  
data.dropna(axis=1, how='all')
#use dict if you want different missing replacements for different columns
data.fillna(data.mean()); data.fillna({1:0, 2:-1}, inplace=True)

#replace bad values - can be a list or a dict
data.replace(-999, np.nan)
print lens[['age', 'age_group']].drop_duplicates()[:10]
data.duplicated().sum()


country_map = {'london': 'England', 'paris': 'France' }
data['country'] = data['city'].map(str.lower).map(country_map)
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)

#bucket the data via quartiles, and turn into dummy variables
quantiles = pd.qcut(cdystonia.age, 4)
pd.get_dummies(quantiles).head(10)



# vectorized if/then 
tweetdf['tweet_type']=np.where(tweetdf['retweeted_status'].notnull(), 'RT', np.where(tweetdf['in_reply_to_status_id_str'].notnull(), 'Reply', 'Other') ) 

#string manipulation - add regex
words = loc.split(',')
country = words[-1].strip().capitalize()
" ".join(collocation).replace('&lt;','<').replace('&gt;','>')


#randomize row order
new_order = np.random.permutation(len(segments))
segments.take(new_order).head()


#apply/map/applymap
#apply - summary of series
#map - element wise on a series
#applymap - element wise on a dataframe (see section5) 

#---------------------------------------------------------------------------
#3. Merge  tables

something for concat


#---------------------------------------------------------------------------
#4. Aggrgations, group by and stack/unstack behaviour, reindex etc

#Index and Multi-index - add/remove from existing df
movies.set_index(['movie_id','release_date'], inplace=True)
movies.swaplevel('movie_id','release_date')
movies.reset_index(inplace=True, level = 1) -- turn the indexed columns into normal columns
#filtering also works on either axis...
movies.query('movie_id==1')
movies.query('release_date=="01-Jan-1995"')






#iterating over groups (e.g. produce a separate analysis, set of charts)
for name, group in movie_ratings.groupby('title'):
    print name
    # do something on the group - could be something different for each group
    group['newcol'] = 1*2 
    print group

#or, without the iterator...
group_A  = grouped['A']

decade_mean = data.groupby(decade).score.mean()
cdystonia_grouped.mean().add_suffix('_mean').head()
.first()
.last()




tips=pd.read_csv('tips.csv')
tips['tips_pct'] = tips['tip']/tips['total_bill']


#groupby functions - use agg (or sum(),first() as summary functions)

#agg - returns scalars
tips.groupby(['sex','smoker'])['tips_pct'].agg('mean').add_suffix('_mean')
tips.groupby(['sex','smoker'])['tips_pct'].agg([('Ave','mean'),('Stdev','std')])

#transform - to do master detail aggregations (i.e. glue agg functions to orig data)
df = pd.DataFrame({'Color': 'Red Red Blue'.split(), 'Value': [100, 150, 50]})
df['Counts'] = df.groupby(['Color']).transform('count')


#apply - generic ufunc to create own custom functions
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(groupkey).apply(fill_mean)
#alternatively - pass predefined missing values to your groups
fill_values = {'East': 0.5, 'West' : 0.9 }
fill_funct = lambda f: g.fillna(fill_values[g.name])
data.groupby(groupkey).apply(fill_func)





#---------------------------------------------------------------------------
#5. Analysis 



#apply a %AGES format to the entire table
from __future__ import division
percentages = pd.crosstab(tweetdf.tweet_day, tweetdf.tweet_type).apply(lambda r: r/r.sum()*100, axis=1)
format = lambda x: '%.1f' % x
percentages.applymap(format)


#Summary Stats -----
#describe - pass arg percentiles=[0.1,0.95] etc to overide defaults
data[['score', 'runtime', 'year', 'votes']].describe()

#running cumulative sum of each series
df.cumsum(); df.cumprod()
#return index (itegers or values)  of row where max/min values are found
df.idmax(); df.idmin(); df.argmin();df.argmax();
#diff/pct_change -  periods =1 (default)
df.diff(); df.pct_change()

#corr/cov - 2 approaches a) calc all across dataframe b) calc a pair of series
# 'method' arg can be pearson, kendall, spearman
df.corr(); df.series1.cov(df.series2)


#Quantile analysis - careful about how the quartile is being passed to groupby
# I think you need to specify the field it is being applied to
quartiles = pd.cut(tips.total_bill, 4)
groups = tips.total_bill.groupby(quartiles).mean()
#equally spaced quartiles
qquartiles = pd.qcut(tips.total_bill, 4)
quartiles.describe()
#or, define explicit cuts manually
bins = np.array([0,1,10,100,1000])
quartiles = pd.cut(tips.total_bill, bins)


#group - weighted averages

df = pd.DataFrame({'category': ['a','a','b','b','b','a'], 'value':np.random.randn(6), 
                   'weights':np.random.rand(6)})
weighted_ave = lambda g: np.average(g.value, weights=g.weights)
groups =  df.groupby('category').apply(weighted_ave)

#pivot table that does counts
pivoted = lens.pivot_table(rows=['movie_id', 'title'], cols=['sex'], values='rating', fill_value=0, aggfunc=np.size)

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


#get the indices of records where col is > 3
col = data[3]
col[np.abs(col)>3]
#or get values ? 3 from any column
data[np.abs(data) > 3).any(1)]

#---------------------------------------------------------------------------
#5. Graphing - stacked bar hist, pie, 

#required for notebook usage
%matplotlib inline

#note - pies can be done with pandas as of 0.14
plt.pyplot.pie(tweetdf.tweet_type.value_counts())

crosstabs = pd.crosstab(tweetdf.tweet_day, tweetdf.tweet_type)
crosstabs.plot(kind='bar', stacked=True)
df['col'].hist(bins=25)

#splom - diagonal can be 'hist'
pd.scatter_matrix(trans_data, diagonal = 'kde', color = 'k', alpha=0.3)

#trellis
tips_data = pd.read_csv('tips.csv')
import pandas.tools.rplot as rplot
plt.figure()
plot = rplot.RPlot(tips_data, x='total_bill', y='tip')
plot.add(rplot.TrellisGrid(['sex', 'smoker']))
plot.add(rplot.GeomHistogram())
plot.render(plt.gcf())

#---------------------------------------------------------------------------
# Stats - from scipy


#some useful functions
np.random.randn(4, 3)
years = range(1880,2011)

shape()
#check that series values are close to but not exactly 1
np.allclose(x, 1) 



#Numeric conversion of Dataframe columns
movies.convert_objects(convert_numeric=True, convert_dates=False)
#memory management - cast the default 64bit float as 32 bit
a.astype(float32)



#------------------------------------------------------------------------------
# List, Set, Tuple and Dict operations

set.union, set.intersection
set(list) -- make a list unique 




#==============================================================================
# Text wrangling    
#==============================================================================

#regex
import re
text = """Dave dave@googlemail.com"""

Steve steve@hotmail.co.uk
Rob ROB@GMAIL.COM
"""

pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2-4}'

regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)