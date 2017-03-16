import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, linear_model
import datetime
import os

os.system('clear')

"""This method is purely for checking that the logic works"""
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# Read in csv table
df = pd.read_csv('nbascraper/command_results.csv')

# Change the string date data in df to datetime format
df['game_date'] = pd.to_datetime(df['game_date'],format='%Y-%m-%d')

df['home_fgp'] = df['home_fg'] / df['home_fga']
df['away_fgp'] = df['away_fg'] / df['away_fga']

df['home_3pp'] = df['home_3p'] / df['home_3pa']
df['away_3pp'] = df['away_3p'] / df['away_3pa']

df['home_ftp'] = df['home_ft'] / df['home_fta']
df['away_ftp'] = df['away_ft'] / df['away_fta']

df.drop(['home_fg','away_fg','home_fga','away_fga','home_3p','away_3p','home_3pa','away_3pa','home_ft','away_ft','home_fta','away_fta','home_ts','away_ts','home_efg','away_efg'],axis=1,inplace=True)

column_list = df.drop(['home_score','away_score','spread'],axis=1).columns

""" Set date after which you don't want to use data to train classifier """
cutoff_date = datetime.date(2015, 8, 1)
""" Creating prediction dataset """
"""*********************************************************************
training_set = df[df.game_date < cutoff_date].set_index('game_date')

model_columns = df.drop(['home_team','away_team','game_date','home_score','away_score','spread'],axis=1).columns

model_set = pd.DataFrame()

model_set['fgp'] = df.home_fgp - df.away_fgp
model_set['3pp'] = df.home_3pp - df.away_3pp
model_set['ftp'] = df.home_ftp - df.away_ftp
model_set['orb'] = df.home_orb - df.away_orb
model_set['drb'] = df.home_drb - df.away_drb
model_set['ast'] = df.home_ast - df.away_ast
model_set['stl'] = df.home_stl - df.away_stl
model_set['blk'] = df.home_blk - df.away_blk
model_set['tov'] = df.home_tov - df.away_tov
model_set['pf'] = df.home_pf - df.away_pf
model_set['spread'] = df.home_score - df.away_score
model_set['game_date'] = df.game_date

# X = model_set.drop(['spread'],axis=1)
X = pd.DataFrame(columns = model_columns)
X[model_columns] = training_set[model_columns]

y = np.array(training_set.home_score - training_set.away_score)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf = svm.SVR(kernel='linear')#linear_model.LogisticRegression(random_state=42)

clf = clf.fit(X_train,y_train)
print clf.score(X_test,y_test)


print clf.predict(X_test)
print y_test
*************************************************************************"""
""" Testing for 2015-2016 Season"""

# Create dataset to be used for prediction
predicting_set = df[df.game_date >= cutoff_date].set_index('game_date')
predicting_set = predicting_set.sort_index()    # Sort ascending by index

del df

team_list = predicting_set.home_team
team_list.drop_duplicates(inplace=True)
team_list = list(team_list)

averages_columns = ['game_count','ast','blk','tov','pf','stl','orb','drb','fgp','ftp','3pp']
averages = pd.DataFrame(columns=averages_columns,index=team_list)
averages.fillna(0,inplace=True)

totals_columns = ['game_count','ast','blk','tov','pf','stl','orb','drb','fgp','ftp','3pp']
totals = pd.DataFrame(columns=totals_columns,index=team_list)
totals.fillna(0,inplace=True)

game_schedule = predicting_set[['home_team','away_team','spread','home_score','away_score']]
game_schedule.reset_index(level=0,inplace=True)
stats_schedule = predicting_set[['home_team','away_team','home_ast','home_blk','home_tov','home_pf','home_stl','home_orb','home_drb','home_fgp','home_ftp','home_3pp','away_ast','away_blk','away_tov','away_pf','away_stl','away_orb','away_drb','away_fgp','away_ftp','away_3pp']]
stats_schedule.reset_index(level=0,inplace=True)

for game in range(163):#len(game_schedule.index)):
	home_games = averages.loc[game_schedule.loc[game,'home_team'],'game_count']
	away_games = averages.loc[game_schedule.loc[game,'away_team'],'game_count']

	# home_game_stats = stats_schedule.loc[game,['home_ast','home_blk','home_tov','home_pf','home_stl','home_orb','home_drb','home_fgp','home_ftp','home_3pp']]
	# away_game_stats = stats_schedule.loc[game,['away_ast','away_blk','away_tov','away_pf','away_stl','away_orb','away_drb','away_fgp','away_ftp','away_3pp']]

	# home_stats = averages.loc[game_schedule.loc[game,'home_team'],averages_columns[1:]] * home_games
	# away_stats = averages.loc[game_schedule.loc[game,'away_team'],averages_columns[1:]] * away_games

	# print totals.loc[game_schedule.loc[game,'home_team'],'ast'], float(stats_schedule.loc[game,['home_ast']])
	totals.loc[game_schedule.loc[game,'home_team'],'ast'] += float(stats_schedule.loc[game,['home_ast']])
	totals.loc[game_schedule.loc[game,'home_team'],'blk'] += float(stats_schedule.loc[game,['home_blk']])
	totals.loc[game_schedule.loc[game,'home_team'],'tov'] += float(stats_schedule.loc[game,['home_tov']])
	totals.loc[game_schedule.loc[game,'home_team'],'pf'] += float(stats_schedule.loc[game,['home_pf']])
	totals.loc[game_schedule.loc[game,'home_team'],'stl'] += float(stats_schedule.loc[game,['home_stl']])
	totals.loc[game_schedule.loc[game,'home_team'],'orb'] += float(stats_schedule.loc[game,['home_orb']])
	totals.loc[game_schedule.loc[game,'home_team'],'drb'] += float(stats_schedule.loc[game,['home_drb']])
	totals.loc[game_schedule.loc[game,'home_team'],'fgp'] += float(stats_schedule.loc[game,['home_fgp']])
	totals.loc[game_schedule.loc[game,'home_team'],'ftp'] += float(stats_schedule.loc[game,['home_ftp']])
	totals.loc[game_schedule.loc[game,'home_team'],'3pp'] += float(stats_schedule.loc[game,['home_3pp']])

	totals.loc[game_schedule.loc[game,'away_team'],'ast'] += float(stats_schedule.loc[game,['away_ast']])
	totals.loc[game_schedule.loc[game,'away_team'],'blk'] += float(stats_schedule.loc[game,['away_blk']])
	totals.loc[game_schedule.loc[game,'away_team'],'tov'] += float(stats_schedule.loc[game,['away_tov']])
	totals.loc[game_schedule.loc[game,'away_team'],'pf'] += float(stats_schedule.loc[game,['away_pf']])
	totals.loc[game_schedule.loc[game,'away_team'],'stl'] += float(stats_schedule.loc[game,['away_stl']])
	totals.loc[game_schedule.loc[game,'away_team'],'orb'] += float(stats_schedule.loc[game,['away_orb']])
	totals.loc[game_schedule.loc[game,'away_team'],'drb'] += float(stats_schedule.loc[game,['away_drb']])
	totals.loc[game_schedule.loc[game,'away_team'],'fgp'] += float(stats_schedule.loc[game,['away_fgp']])
	totals.loc[game_schedule.loc[game,'away_team'],'ftp'] += float(stats_schedule.loc[game,['away_ftp']])
	totals.loc[game_schedule.loc[game,'away_team'],'3pp'] += float(stats_schedule.loc[game,['away_3pp']])

	# print game_schedule.loc[game,'home_team']
	# if game == 11:
	# 	print home_game_stats['home_ast']
	# 	print away_game_stats

	# averages.loc[game_schedule.loc[game,'home_team'],'game_count'] += 1
	# averages.loc[game_schedule.loc[game,'away_team'],'game_count'] += 1
	totals.loc[game_schedule.loc[game,'home_team'],'game_count'] += 1
	totals.loc[game_schedule.loc[game,'away_team'],'game_count'] += 1

	# home_stats = home_stats / averages.loc[game_schedule.loc[game,'home_team'],'game_count']
	# away_stats = away_stats / averages.loc[game_schedule.loc[game,'away_team'],'game_count']

	# averages.loc[game_schedule.loc[game,'home_team'],averages_columns[1:]] = home_stats
	# averages.loc[game_schedule.loc[game,'away_team'],averages_columns[1:]] = away_stats

print totals.loc['MIN',['ast','blk','stl','tov','orb','drb']]
# date_range = pd.date_range(game_schedule.index.min().date(),game_schedule.index.max().date())
# print date_range