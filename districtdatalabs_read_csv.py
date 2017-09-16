# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:19:36 2016

@author: stevegoodman
"""
from __future__ import print_function
import csv
from collections import Counter
path= '/Users/stevegoodman/downloads/TechCrunchcontinentalUSA.csv'
def read_csv():
    with open('/Users/stevegoodman/downloads/TechCrunchcontinentalUSA.csv', 'rU') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

class BetterReader(object):
    def __init__(self, path):
        self.path = path
        self._length = None
        self._counter = None
    def __iter__(self):
        self._length = 0
        self._counter = Counter()
        with open(self.path, 'rU') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._length += 1
                self._counter[row['company']] += 1
                yield row
    def __len__(self):
        if self._length is None:
            for row in self: continue
        return self._length
    
    @property
    def counter(self):
        if self._counter is None:
            for row in self: continue
        return self._counter
    @property
    def companies(self):
        return self.counter.keys()
        
    def reset(self):
        self._length = None
        self._counter = None
        


if __name__ == "__main__":
#==============================================================================
#     for ix, row in enumerate(read_csv()):
#         if ix > 10:
#             break;
#         print(row['city'])
#==============================================================================
    reader = BetterReader(path)
    print(reader.companies)
    print(len(reader))
    print(reader.counter)
    