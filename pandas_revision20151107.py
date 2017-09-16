55# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:24:39 2015

@author: stevegoodman
"""

import numpy as np
import pandas as pd
from __future__ import print_function
from __future__ import division
from pandas import DataFrame, Series




data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
}
frame1 = DataFrame(data)


#Apply - reduction over an axis

#INCORRECT - works on DF not series
frame1['pop'].apply(lambda x: x.max() - x.min() )

frame = DataFrame(np.random.random_integers(1,100,(4, 3)), 
                  columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])

#CORRECT
frame.apply(lambda x: x.max() - x.min() )


##Map elementwise transformation over a Series


### ApplyMap Elmentwise transformation over a DF



# Summary functions

frame.mean(axis=1) # agg accross the colums
frame.idxmax()
frame.cumsum()
frame.describe()

pct = frame.pct_change()

#format to 2dp
pct.applymap(lambda x: '%.2f' % x)

#newer python version
pct.applymap(lambda x: '{0:.2%}'.format(x))



#value counts - calculate across all columns simultanesiously
# fillna replaces nans with 0

data = DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
                  
result = data.apply(pd.value_counts).fillna(0)]
#FILL IN AVG OF A SERIES
data.fillna(data.mean())
 
df = pd.DataFrame([[1, np.nan, 2],[2 ,3, 5], [np.nan, 4,6]])
df.dropna(axis='columns') #alias for axis =1
df.dropna(axis='rows') #alias for axis=0)
df.dropna(thresh=3)
 
# 
#INDEXING
#Note that with slicing explict indexing (labels) the range a:z is inclusive
#But with implicit integer indexes 1:10 range exlcudes last value
#
 
area = pd.Series({'California': 423967, 'Texas': 695662,
                      'New York': 141297, 'Florida': 170312,
                      'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                     'New York': 19651127, 'Florida': 19552860,
                     'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data
data.loc[:,'area']
data.iloc[:,0]
data.ix[1,'area']

#Build a multi-index from existing column names
pop_flat.set_index(['state', 'year'])
 
#turn index into a column(s)
pop_flat.reset_index()
 
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
 
 
#==============================================================================
# Concat and merge by Vanderplas
#==============================================================================
# rank US states & territories by their 2010 population density.

!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv
!curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv

pop = pd.read_csv('state-population.csv')
areas = pd.read_csv('state-areas.csv')
abbr = pd.read_csv('state-abbrevs.csv')


merged= pd.merge(pop, abbr, how='outer', left_on='state/region', right_on='abbreviation')
merged.drop('abbreviation', axis='columns', inplace=True)
merged.isnull().any()

#infill the missing state fields where abbreivations are either PR or USA
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Peuto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()

final = pd.merge(merged, areas, on='state', how='left')
final.head()
final.dropna(inplace=True)

data2010 = final[final['year']==2010]
data2010 = final.query('year==2010 and ages=="total"'   )


#==============================================================================
# Group by from VDP when I had no internet connection to download the planets
#==============================================================================

flights = pd.read_csv('/Users/stevegoodman/Downloads/flights.csv')

flights.describe()

flights.groupby('month')['dep_delay'].mean()

flights.groupby('month').aggregate(['min', max])
flights.groupby('month').aggregate({'air_time':'min', 'distance': 'max'})

#==============================================================================
#  Group by From Vanderplas
#==============================================================================

import seaborn as sns
planets = sns.load_dataset('planets')

planets.groupby('method').median()

#can make the index a column, but it depends what your returning
#describe() won't do it
planres = planets.groupby('method',as_index=False)['mass'].mean()

planets.groupby('method').groups
#New use of a filter...

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                    columns = ['key', 'data1', 'data2'])

def filter_func(x):
        return x['data2'].std() > 4
df.groupby('key').std()

df.groupby('key').filter(filter_func)


planets.groupby(['method', planets.year//10*10])['number'].sum().unstack()


#Nice pattern - group by part of date e.g. month/year of date..


tulsa.groupby(tulsa['startdt'].dt.month).PRODUCTPRICE.mean()
['PRODUCTPRICE'].mean()




##
## Transform - assumes either a scalar value is returned and broadcasted (like np.mean)
### or , returns a transformed array of the same size

people = DataFrame(np.random.randn(5, 5), columns=['a', 'b', 'c', 'd', 'e'],
index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])

people.ix[2:3, ['b', 'c']] = np.nan # Add a few NA values

key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()
people.groupby(key).transform(np.mean)

def demean(arr):
    return arr - arr.mean()

demean = people.groupby(key).transform(demean)
demean.groupby(key).mean()

## Apply is more general purpose - no restrictions on what gets returned

tips = pd.read_csv('/Users/stevegoodman/Documents/Dev/tips.csv')
def top(grp, n=5):
    return grp.tip.order(ascending=False)[:n]
    
tips.groupby('smoker').apply(top,2)



### Anothper great pattern group by a binned continuous variable

factor = pd.cut(tips.total_bill, 5)

tips.groupby(factor).tip.mean()
# what percent of bill is the tip?
#Note : apply is better than agg here because
#we want to return just the derived variable
def pct_bill(x):
    return x.tip.sum()/x.total_bill.sum()
tips.groupby(factor).apply(pct_bill)


df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], 'data': np.random.randn(8),
'weights': np.random.rand(8)})

grouped = df.groupby('category')
avg = lambda x: np.average(x['data'],weights=x['weights'] )
grouped.apply(avg)
#this won't work becuase returned value is not a DF
grouped.transform(avg)


### Pivot tables

tips.pivot_table('tip','sex','smoker',margins=True) \
    .applymap(lambda x: '%.2f' % x)

#age by gender - question, how do we dedupe for repeated values? 
pd.crosstab(tulsa.AGE, tulsa.S1)

tulsa2 = tulsa.reset_index().drop_duplicates(subset='RespondentID')
pd.crosstab(tulsa2.AGE, tulsa2.S1)



# FEC example from McKinney

fec = pd.read_csv('/Users/stevegoodman/Documents/Dev/pydata-book-master/ch09/P00000001-ALL.csv')

fec.cand_nm.unique()

parties = {'Bachmann, Michelle': 'Republican', 'Cain, Herman': 'Republican',
'Gingrich, Newt': 'Republican', 'Huntsman, Jon': 'Republican', 'Johnson, Gary Earl': 'Republican', 'McCotter, Thaddeus G': 'Republican', 'Obama, Barack': 'Democrat',
'Paul, Ron': 'Republican',
'Pawlenty, Timothy': 'Republican',
'Perry, Rick': 'Republican',
"Roemer, Charles E. 'Buddy' III": 'Republican', 'Romney, Mitt': 'Republican',
'Santorum, Rick': 'Republican'}

fec['party'] = fec.cand_nm.map(parties)
fec.groupby('party').size()
fec['party'].value_counts()

fec[fec.contb_receipt_amt >0]['party'].value_counts()

#contributions by top occupations
fec.groupby('contbr_occupation').contb_receipt_amt.sum().sort_values(ascending=False)[:10]

#map some similar occupations onto just 1 job
occ_map = {'INFORMATION REQUESTED PER BEST EFFORTS' : 'INFORMATION REQUESTED'}

fec['contbr_occupation']  = fec['contbr_occupation'].map(lambda x: occ_map.get(x,x))


#==============================================================================
# Titanic data (vanderplas/portilla)
#==============================================================================

import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

sns.factorplot('sex',data=titanic)
sns.factorplot('sex',data=titanic, hue='pclass')
sns.factorplot('pclass',data=titanic, hue='sex')

#split out children as a separate category
#NOTE :::: WORK OUT WHY apply (axis=1)
def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex
titanic['person'] = titanic[['age', 'sex']].apply(male_female_child, axis=1)

mfc.value_counts()
titanic['age'].mean()
titanic['age'].median()

titanic['age'].hist(bins=70)

fig = sns.FacetGrid(titanic, hue='sex', aspect=4)
fig.map(sns.kdeplot, 'age', shade=True)
oldest = titanic['age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

deck2 = titanic['deck'].dropna()

# Survivor analysis
titanic['survivor'] = titanic['survived'].map({0:'No', 1:'Yes'})
titanic['survivor'].value_counts()
sns.factorplot('survivor', data=titanic, palette='Set1')
sns.factorplot('pclass','survived', hue='person',data=titanic)

titanic.person2 = titanic[['age', 'sex']].apply(male_female_child, axis=1)


#==============================================================================
# Voter data - Jose portilla
#==============================================================================

import requests
from StringIO import StringIO
url = "http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv"
source = requests.get(url).text
poll_data= StringIO(source)

poll = pd.read_csv(poll_data)

sns.factorplot('Affiliation', hue='Population', data = poll)

avg = DataFrame(poll.mean())
std = DataFrame(poll.std())
avg.drop('Number of Observations',axis=0, inplace=True)
std.drop('Number of Observations',axis=0, inplace=True)

avg.plot.bar(legend=False,yerr=std)

pollavg = pd.concat([avg,std], axis=1)
pollavg.columns=['avg','std']

#reorder so time goes LtoR
poll[::-1].plot(x='End Date',y=['Romney','Obama','Undecided'], marker='o',linestyle='')



#Plot differences between the two candidates
poll['diff'] = (poll['Obama'] - poll['Romney'])/100

poll2 = poll.groupby(['Start Date'],as_index=False).mean()

poll.plot('Start Date','diff',figsize=(12,4),marker='o',linestyle='-',color='purple')

### OR New pandas style with plot accessor attribute
poll.plot.line('Start Date','diff',figsize=(12,4),marker='o',linestyle='-',color='purple')


#==============================================================================
# Date indexing - extracting a slice e.g. by month or year.
#==============================================================================

dateframe = DataFrame( np.random.randn(6,4), columns=['a','b','c','d'], index=pd.date_range('20150101',periods=6, freq='M') )
dateframe.loc['201501':'201503']

dateframe.sample(n=2)   

#==============================================================================
# More advanced masks
#==============================================================================
df2 = pd.DataFrame({'a' : ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b' : ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c' : np.random.randn(7)})
# only want 'two' or 'three'
criterion = df2['a'].map(lambda x: x.startswith('t'))
df2[criterion]


#==============================================================================
# C7 Data wrangling - Mkinney
#==============================================================================

#Binning cont variables, and change the bin labels

ages = np.random.random_integers(16,75,100)
bins = [0,18,30,50,65]
cats = pd.cut(ages, bins)
generations = ['tweeners','millinials','genX','boomers']
cats= pd.cut(ages, bins, labels=generations)
 
#Categorical vars would work for likert scale data as in Tulsa
#although the copy of tulsa I have has used text rather than the underlying scale
 
 
 
 
#filter outliers
np.random.seed(12345)
data= DataFrame(np.random.randn(1000, 4)) 

data.describe()

data[(data >3).any(1)]



######## STRING processing
val= 'a, b,  guido'
pieces = val.split(',')
pieces = [x.strip() for x in pieces]
":".join(pieces)
'guido' in val
val.find('giiiuido')
val.count(',')
val.capitalize()

import re
text = "foo    bar\t baz   \tqux"
#split on whitespace
re.split('\s+',text)

text = """Dave dave@google.com Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
print ( regex.sub('REDACTED', text) )


#from jose s workbook
test_phrase = r'sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd'

test_patterns = [ '[sd]',    # either s or d
            's[sd]+']   # s followed by one or more s or d
            

re.search('s[sd]+', test_phrase).start()

#s followed by 1+ d or s
re.findall('s[sd]+', test_phrase)

#s followed by 0+ d/s
re.findall('s[sd]*', test_phrase)

#s followed by 0,1 d/s
re.findall('s[sd]?', test_phrase)

#s followed by 2 or 3 d
re.findall('s[d]{2,3}', test_phrase)


test_phrase2 = 'This is a string! But it has punctutation. How can we remove it?'
#remove punctuation

re.findall(r'[^-!?. ]+', test_phrase2)


test_phrase3 = 'This is a string with some numbers 1233 and a symbol #hashtag'

test_patterns=[ r'\d+', # sequence of digits
                r'\D+', # sequence of non-digits
                r'\s+', # sequence of whitespace
                r'\S+', # sequence of non-whitespace
                r'\w+', # alphanumeric characters
                r'\W+', # non-alphanumeric
                ]
re.findall(r'\d+', test_phrase3) # sequence of digits
re.findall(r'\D+', test_phrase3) # sequence of non digits
re.findall(r'\s+', test_phrase3) # sequence of whitespace
re.findall(r'\S+', test_phrase3) # sequence of nonwhitespace
re.findall(r'\w+', test_phrase3) # sequence of alphanum
re.findall(r'\W+', test_phrase3) # sequence of nonalphanum

#Except for control characters, (+ ? . * ^ $ ( ) [ ] { } | \), all characters match themselves.
# You can escape a control character by preceding it with a backslash.
# In which case you should also use raw strings otherwise need double backslash
#==============================================================================
# CAtegoricals
#==============================================================================

s = pd.Series(['a', 'b', 'c', 'a'], dtype="category")

#this is agreat pattern for auto creation of e.g. age ranges
df = pd.DataFrame({'value': np.random.randint(0, 100, 20)})
labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]
df['group'] = pd.cut(df.value, range(0, 105, 10), right=False, labels=labels)
df.head(10)
 
 
 

 
 
#==============================================================================
# McKinney's C2 - intro examples
#==============================================================================
path = '/Users/stevegoodman/Documents/Dev/pydata-book-master'
 
import json
with open(path+'/ch02/usagov_bitly_data2012-03-16-1331923249.txt','rb') as f:
    records = [json.loads(line) for line in f]
#records is a list of dicts
records[0]['c']

time_zones = Series( [rec['tz'] for rec in records if 'tz' in rec] )
time_zones2 = [rec['tz'] for rec in records if 'tz' in rec]

#COUNT timezones - newish way in pure python
from collections import Counter

cntr = Counter(time_zones2)

cntr.most_common(10)
# or pandas
time_zones.value_counts()


 # Heres a useful patternFind the ten most common words in Hamlet
>>> import re
>>> words = re.findall(r'\w+', open('hamlet.txt').read().lower())
>>> Counter(words).most_common(10)

frame = DataFrame(records)
#want to know how many have missings - note empty string '' is not same as missing
frame.tz.value_counts(dropna=False)

#see counts of browser details

browsers = Series([x.split()[0] for x in frame['a'].dropna()])

##More modern approach to the above
browser2 = Series(frame['a'] \
            .str.split(' ') \
            .str.get(0))


operating_system = np.where(frame['a'].str.contains('Windows'),
'Windows', 'Not Windows')

frame=frame[frame.notnull()]

agg_counts = frame.groupby(['tz', operating_system]) \
            .size() \
            .unstack() \
            .fillna(0)

agg_counts[:10]

#use to sort in ascending order
#argsort used just to get the indices of the sort by summed columns
## OF course- a simpler way would just be to derive a 'total' column then sort by it
## take the index of 
indexer = agg_counts.sum(axis=1).argsort()
indexer[:10]
count_subset = agg_counts.take(indexer)[-10:]
count_subset
import seaborn as sns
count_subset.plot(kind='barh', stacked=True)

#stacked - clculate proportion of row of each of the two columns
#div(x) is a df elementwise division by x

normed_subset = count_subset.div(count_subset.sum(1), axis=0)



#==============================================================================
# Jake VP -Birthrate data
#==============================================================================
!curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/births.csv

births = pd.read_csv('births.csv')
births.head()

births['decade'] = (10* (births['year'] //10))

births.pivot_table('births',index='decade', columns='gender', aggfunc='sum')

#visualise the trend...
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
births.pivot_table('births',index='decade', columns='gender', aggfunc='sum').plot()
plt.ylabel('births')

## Clean up some wrong dates and create a date index?

#rowwise deletion - remove dates that are string=="null"" or 99
#

#==============================================================================
#==============================================================================
# # Got to remember working off of a copy
#==============================================================================
#==============================================================================

births['date'] = pd.to_datetime(births.day +births.month.astype('string')+births.year.astype('string'), 
                      format="%d%m%Y", errors='coerce' )
                      
                      
#births_clean = births[(births.day!='null') & (births.day!='99')] 


# Q What happens to the indeix  if the date is wrong?
# A = cant create a datetime with wrong date to errors=coerce will create NaT
# and hence ....will return indices with integers rather than dates...hmm...
# (errors=ignore would return the input which is no good either

#filter out NaT


births.index=births.date
births['dayofweek']=births.index.dayofweek

births.pivot_table('births','month',aggfunc='sum').plot()


#==============================================================================
#J VDP - string operations 
#==============================================================================
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
'Eric Idle', 'Terry Jones', 'Michael Palin'])

monte.str.lower()
monte.str.split().str.get(0)

monte.str.match('[a-zA-Z]+')
monte.str.extract('