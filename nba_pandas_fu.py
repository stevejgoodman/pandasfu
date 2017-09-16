#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:04:41 2017
Tidy data example
https://tomaugspurger.github.io/modern-5-tidy

@author: stevegoodman
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    import prep # noqa

pd.options.display.max_rows = 10
sns.set(style='ticks', context='talk')

fp = './Documents/dev/nba.csv'
tables = pd.read_html("http://www.basketball-reference.com/leagues/NBA_2016_games.html")
games = tables[0]
games.to_csv(fp)
else:
    games = pd.read_csv(fp)
games.head()

column_names = {'Date': 'date', 'Start (ET)': 'start',
                'Unamed: 2': 'box', 'Visitor/Neutral': 'away_team', 
                'PTS': 'away_points', 'Home/Neutral': 'home_team',
                'PTS.1': 'home_points', 'Unamed: 7': 'n_ot'}

games = (games.rename(columns=column_names)
    .dropna(thresh=4)
    [['date', 'away_team', 'away_points', 'home_team', 'home_points']]
    .assign(date=lambda x: pd.to_datetime(x['date'], format='%a, %b %d, %Y'))
    .set_index('date', append=True)
    .rename_axis(["game_id", "date"])
    .sort_index())
games.head()

#Calc rest between games.
teams = pd.melt(games.reset_index(), id_vars=['game_id','date'], value_vars=['home_team','away_team'])
teams = teams.sort_values(['value','date'])
teams['rest'] = teams.groupby(['value']).date.diff().dt.days - 1

by_game = pd.pivot_table(teams, index= ['game_id','date'], columns='variable', values='rest')

df = pd.concat([games, by_game], axis=1)
df = df.rename(columns={'away_team': 'away_rest',
                              'home_team': 'home_rest'})

sns.set(style='ticks', context='paper')
g = sns.FacetGrid(teams, col='value', col_wrap=5, hue='value', size=2)
g.map(sns.barplot, 'variable', 'rest')

df['home_win'] = df['home_points'] > df['away_points']
df['rest_spread'] = df['home_rest'] - df['away_rest']