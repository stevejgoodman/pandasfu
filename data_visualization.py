# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:17:58 2014
Visualisation
@author: stevegoodman
"""


#==============================================================================
# GGPLOT
#==============================================================================
- switch to mathplotlib OOP (ax instead of plt)
- prevent copying data
- qplot
- improved color selection
- geom_rect
- scale_colour_brewer
- better geom_bar defaults
- theme_seaborn
- scale_x_discrete
- scale_y_discrete
- jitter is now just a special case of geom_point
- basic ggplotrc


from ggplot import *
#Plot a series
qplot(meat.beef)


#facets

p = ggplot(aes(x='price'), data=diamonds)
p + geom_histogram() + facet_wrap("cut")

#custom axis breaks and labelling
ggplot(mtcars, aes('mpg', 'qsec')) + \
  geom_point(colour='steelblue') + \
  scale_x_continuous(breaks=[10,20,30],  \
  labels=["horrible", "ok", "awesome"]) + \
  geom_point(color = "red") + ggtitle("Cars") + xlab("Weight") + ylab("MPG")
#dates
ggplot(meat, aes('date','beef')) + \
    geom_line(color='black') + \
    scale_x_date(breaks=date_breaks('7 years'), labels='%b %Y') + \
    scale_y_continuous(labels='comma')
    
    
    
#==============================================================================
# Matplotlib - use the Harvard examples for class profiling
#==============================================================================
