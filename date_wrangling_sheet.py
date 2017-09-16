# -*- coding: utf-8 -*-
"""
date_wrangling (use examples from data_wrangling_sheet.py and movielensexample)

@author: stevegoodman
"""

#---------------------------------------------------------------------------
# Date time manipulation
#(Q: how about conversion from SQL queries?)

# Autodetect with a custom library- single element, (Pandas uses by default)
from dateutil.parser import parse
parse(segments.st_time.ix[0], dayfirst=True)

#or supply a format with built in functionality, and vectorise it to a series
segments.st_time.apply(lambda d: datetime.strptime(d, '%m/%d/%y %H:%M'))

#or use pandas builtin to apply to a series
pd.to_datetime(segments.st_time)
tweetdf['tweet_day'] = tweetdf['tweet_date'].map(pd.Timestamp.date)

#specify as part of the load code (need to indicate columns in a list)
movies = pd.read_csv('/Users/stevegoodman/Documents/Dev/ml-100k/u.item', sep='|', index_col=0, names=m_cols, usecols=range(5), parse_dates=[2])

#unix timestamp
datetime.datetime.fromtimestamp(float(os.path.getmtime("FILE"))).strftime("%B %d, %Y")

#vectorise date functions - weekday 
movies.release_date.apply(lambda d: datetime.datetime.weekday(d))


# appy timestamps to a series
#force series to be a datetime object - unrecognisable dates will be NaT
#without it, will be converted to a generic object
#infer will base it on the first row (mostly to provide a speedip?)
pd.to_datetime([1, '1970'], coerce=True, infer_datetime_format=True)


#Question - how to select on dates when datecol is not an index?
# numpy datetime64 seems to inherit the functions/fields of datetime
from datetime import datetime 

# Vectorised date operations and filtering 

def get_year(mydate):
    return mydate.year


#should this not be 'map' as it works element wise?     
movies['release_year'] = movies.release_date.apply(get_year)
movies['release_year'] == 1996

    
# can be list or dict - makes more sense with strings?
movies['release_year'].isin(range(1996,2000) )


#subset the rows (note that the where method will return something similar
#but with a similar shape to the original DF with NaN elements for non-matching rows
movies['release_year'] > 1996

movies[(movies['release_date'] >'1996-03-08') & (movies['release_date'] < '1996-03-10')]
#spec 'index' to query by the row index, can use 'and/or/not' and 'in/not in'
#see use cases in pandas doc that illustrates how same query can be passed 
#multiple DFs that have common columns, in a single command

movies.query('(release_date > "1996-03-08") & (release_date < "1996-03-10")')

#format date string representation
mydate.strftime('%Y%m%d')


#timedeltas and date arithmatic 


#Rem - look at multi indexing to compress customerid/transdate transaction datasets
# possibly use for crosstabs,plots etc

#OR just leave to group by indexing
#first, last sum, time dif between first/last etc




#==============================================================================
#Periods 
#==============================================================================

p= pd.Period(2007, freq='A-DEC')

#if DF has a DatetimeIndex is a transform a date column into something based e.g. on quarters
pts = movie_ratings.to_period('M')
#and convert it back
pts.to_timestamp()

#from wes' book p 312 doesn't work out of the box...
#unless convert series to a list 
data = pd.read_csv('pydata-book-master/ch08/macrodata.csv')
index= pd.PeriodIndex(year=list(data.year), quarter=list(data.quarter), freq = 'Q-DEC')
data.index =index