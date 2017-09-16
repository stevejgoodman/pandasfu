# -*- coding: utf-8 -*-
"""
Following Wes McK's pycon tutorial and books

Created on Sat May 24 15:35:40 2014

@author: stevegoodman
"""



#==============================================================================
# Biths data
#==============================================================================
years = range(1880,2011)
pieces=[]
columns=['name','sex','birth']
for year in years:
    os.path.abspath(os.path.join('Users','stevegoodman','Documents','Dev','pydata-book-master','ch02','names','yob%d.txt') % year)    
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

names = pd.concat(pieces, ignore_index=True)
names.index
names
total_births = names.pivot_table('birth', rows='year',cols='sex', aggfunc=sum)
total_births
names.pivot_table('birth', rows='year',cols='sex', aggfunc=sum).plot()
def add_prop(group):
    #integer division floors
    birth = group.birth.astype(float)
    group['prop'] = birth/birth.sum()
    return group

names = names.groupby(['year','sex']).apply(add_prop)
names
names.groupby(['year','sex']).sum()
def get_top1000(group):
        return group.sort_index(by='birth', ascending=False)[:1000]

grouped = names.groupby(['year','sex'])
top1000=grouped.apply(get_top1000)


#applying string formats/transformations - use a dict and pass it to map
#taking care of matching case at the same time
country_map = {'london': 'England', 'paris': 'France' }
data['country'] = data['city'].map(str.lower).map(country_map)



#applying a temporary format to a group by transformation
# see pp258
mapping = {'a':'red','b':'red','c':'blue', 'd':'blue'}
by_column = people.groupby(mapping,axis=1)


# Use seaborn to override matplotlib defaults
import seaborn as sns


#==============================================================================
# Ded Election Comission
#==============================================================================
fec = pd.read_csv('pydata-book-master/ch09/P00000001-ALL.csv')
unique_cands = fec.cand_nm.unique()
#candidate mappings - use a dict trick that will allow non-mapped candidates
#to 'pass through' the mapping without raising a keyerror
#requires 1 extra (lambda) function

parties = {'Bachmann, Michelle' : 'Rep', 'Obama, Barack':'Dem',
'Romney, Mitt':'Rep'}

party_map = lambda x: parties.get(x, x)
fec['party'] = fec.cand_nm.map(party_map)

(fec.contb_receipt_amt > 200000).value_counts()
fec[fec.contb_receipt_amt > 200000]

#restrict data to positive contributions (i.e. remove refunds)
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]

#pivottable of part and occupation then filtr to subset donating at least 2m
#overall (hence sum where axis = 1)

over2mm = by_occupation[by_occupation.sum(1) > 200000 ]
by_occupation = fec[fec.party.isin(['Rep','Dem'])].pivot_table('contb_receipt_amt', columns = 'party', index='contbr_occupation', aggfunc='sum')

#top mmounts
def group_by_top(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.order(ascending=False)[:n]

#cuts
bins = np.array([0,1,10,100,1000,10000,100000,1000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
groups = fec_mrbo.groupby(['cand_nm',labels]).size()
groups.unstack(0)


#This is cool - cal proportions across columns whilst summing by rows
groups = fec_mrbo.groupby(['cand_nm',labels])
bucket_sums = groups.contb_receipt_amt.sum().unstack(0)
#take eack colum in turn and divide by total across columns
normed_sums = bucket_sums.div(bucket_sums.sum(1), axis=0)
#How do I acheive a similar effect with sql window functions?

