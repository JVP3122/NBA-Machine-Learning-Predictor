import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, linear_model
from sklearn.feature_selection import RFE
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
# """*********************************************************************
training_set = df[df.game_date < cutoff_date].set_index('game_date')

model_columns = df.drop(['home_team','away_team','game_date','home_score','away_score','spread'],axis=1).columns

# model_set = pd.DataFrame()

# model_set['fgp'] = df.home_fgp - df.away_fgp
# model_set['3pp'] = df.home_3pp - df.away_3pp
# model_set['ftp'] = df.home_ftp - df.away_ftp
# model_set['orb'] = df.home_orb - df.away_orb
# model_set['drb'] = df.home_drb - df.away_drb
# model_set['ast'] = df.home_ast - df.away_ast
# model_set['stl'] = df.home_stl - df.away_stl
# model_set['blk'] = df.home_blk - df.away_blk
# model_set['tov'] = df.home_tov - df.away_tov
# model_set['pf'] = df.home_pf - df.away_pf
# model_set['spread'] = df.home_score - df.away_score
# model_set['game_date'] = df.game_date

# X = model_set.drop(['spread'],axis=1)
X = pd.DataFrame(columns = model_columns)
X[model_columns] = training_set[model_columns]

y = np.array(np.where(training_set.home_score > training_set.away_score,1,0))

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
# C_vec = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 6.0, 10.0]
# prob_val = 0
# C_val = 0

# for c in C_vec:
# clf = svm.SVR(C=0.03,kernel='linear')#linear_model.LogisticRegression(random_state=42)
# 	# clf = clf.fit(X,y)
# 	clf = clf.fit(X_train,y_train)
# 	prob_result = clf.score(X_test,y_test)
# 	if prob_result > prob_val:
# 	    prob_val = prob_result
#     	C_val = c
# 	print 'C =',c,'Score = ',prob_result

clf = linear_model.LogisticRegression(C=3,random_state=23)
clf = clf.fit(X_train,y_train)
print clf.score(X_test,y_test)

print 'Hit Enter to Continue...'
raw_input()
print clf.predict_proba(X_test)
print y_test

# reg = linear_model.ElasticNet(alpha=.0001,max_iter=1000000000)
# reg = reg.fit(X_train,y_train)
# print reg.score(X_test,y_test)

# *************************************************************************"""

""" Testing for 2015-2016 Season"""
"""*********************************************************************
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

game_schedule = predicting_set[['home_team','away_team','spread','home_score','away_score']]
game_schedule.reset_index(level=0,inplace=True)
stats_schedule = predicting_set[['home_team','away_team','home_ast','home_blk','home_tov','home_pf','home_stl','home_orb','home_drb','home_fgp','home_ftp','home_3pp','away_ast','away_blk','away_tov','away_pf','away_stl','away_orb','away_drb','away_fgp','away_ftp','away_3pp']]
stats_schedule.reset_index(level=0,inplace=True)

result_columns = ['home_team','away_team','vegas_spread','game_spread','home_score','away_score','bet','result_value','win_loss']
result = pd.DataFrame(columns = result_columns)

for game in range(len(game_schedule.index)):
	home_games = averages.loc[game_schedule.loc[game,'home_team'],'game_count']
	away_games = averages.loc[game_schedule.loc[game,'away_team'],'game_count']

	home_stats = averages.loc[game_schedule.loc[game,'home_team'],averages_columns[1:]] * home_games
	away_stats = averages.loc[game_schedule.loc[game,'away_team'],averages_columns[1:]] * away_games

	if home_games > 20 and away_games > 20:
		predictor = pd.DataFrame(columns = model_columns)
		predictor.loc[0,:] = 0
		predictor['away_ast'] = averages.loc[game_schedule.loc[game,'away_team'],'ast']
		predictor['away_blk'] = averages.loc[game_schedule.loc[game,'away_team'],'blk']
		predictor['away_tov'] = averages.loc[game_schedule.loc[game,'away_team'],'tov']
		predictor['away_pf'] = averages.loc[game_schedule.loc[game,'away_team'],'pf']
		predictor['away_orb'] = averages.loc[game_schedule.loc[game,'away_team'],'orb']
		predictor['away_drb'] = averages.loc[game_schedule.loc[game,'away_team'],'drb']
		predictor['away_stl'] = averages.loc[game_schedule.loc[game,'away_team'],'stl']
		predictor['away_fgp'] = averages.loc[game_schedule.loc[game,'away_team'],'fgp']
		predictor['away_ftp'] = averages.loc[game_schedule.loc[game,'away_team'],'ftp']
		predictor['away_3pp'] = averages.loc[game_schedule.loc[game,'away_team'],'3pp']

		predictor['home_ast'] = averages.loc[game_schedule.loc[game,'home_team'],'ast']
		predictor['home_blk'] = averages.loc[game_schedule.loc[game,'home_team'],'blk']
		predictor['home_tov'] = averages.loc[game_schedule.loc[game,'home_team'],'tov']
		predictor['home_pf'] = averages.loc[game_schedule.loc[game,'home_team'],'pf']
		predictor['home_orb'] = averages.loc[game_schedule.loc[game,'home_team'],'orb']
		predictor['home_drb'] = averages.loc[game_schedule.loc[game,'home_team'],'drb']
		predictor['home_stl'] = averages.loc[game_schedule.loc[game,'home_team'],'stl']
		predictor['home_fgp'] = averages.loc[game_schedule.loc[game,'home_team'],'fgp']
		predictor['home_ftp'] = averages.loc[game_schedule.loc[game,'home_team'],'ftp']
		predictor['home_3pp'] = averages.loc[game_schedule.loc[game,'home_team'],'3pp']

		game_result = pd.DataFrame(columns = result_columns)
		game_result.loc[0,:] = 0

		predictor = scaler.transform(predictor)
		game_result['home_team'] = game_schedule.loc[game,'home_team']
		game_result['away_team'] = game_schedule.loc[game,'away_team']
		game_result['vegas_spread'] = game_schedule.loc[game,'spread']
		game_result['home_score'] = game_schedule.loc[game,'home_score']
		game_result['away_score'] = game_schedule.loc[game,'away_score']
		game_result['game_spread'] = rfe.predict(predictor)

		if game_result.loc[0,'game_spread'] < game_result.loc[0,'vegas_spread']:
			game_result.loc[0,'bet'] = 'home'
		else:
			game_result.loc[0,'bet'] = 'away'
		
		if game_result.loc[0,'home_score'] + game_result.loc[0,'vegas_spread'] > game_result.loc[0,'away_score']:
			game_result['result_value'] = 'home'
		elif game_result.loc[0,'home_score'] + game_result.loc[0,'vegas_spread'] < game_result.loc[0,'away_score']:
			game_result.loc[0,'result_value'] = 'away'
		else:
			game_result.loc[0,'result_value'] = 'push'
		
		if (game_result.loc[0,'bet'] == game_result.loc[0,'result_value']):
			game_result.loc[0,'win_loss'] = 1
		elif game_result.loc[0,'result_value'] == 'push':
			game_result.loc[0,'win_loss'] = 0
		else:
			game_result.loc[0,'win_loss'] = -1


		# print game_result
		result = result.append(game_result)

	
	home_stats['ast'] += stats_schedule.loc[game,['home_ast']]
	home_stats['blk'] += stats_schedule.loc[game,['home_blk']]
	home_stats['tov'] += stats_schedule.loc[game,['home_tov']]
	home_stats['pf'] += stats_schedule.loc[game,['home_pf']]
	home_stats['stl'] += stats_schedule.loc[game,['home_stl']]
	home_stats['orb'] += stats_schedule.loc[game,['home_orb']]
	home_stats['drb'] += stats_schedule.loc[game,['home_drb']]
	home_stats['fgp'] += stats_schedule.loc[game,['home_fgp']]
	home_stats['ftp'] += stats_schedule.loc[game,['home_ftp']]
	home_stats['3pp'] += stats_schedule.loc[game,['home_3pp']]

	away_stats['ast'] += stats_schedule.loc[game,['away_ast']]
	away_stats['blk'] += stats_schedule.loc[game,['away_blk']]
	away_stats['tov'] += stats_schedule.loc[game,['away_tov']]
	away_stats['pf'] += stats_schedule.loc[game,['away_pf']]
	away_stats['stl'] += stats_schedule.loc[game,['away_stl']]
	away_stats['orb'] += stats_schedule.loc[game,['away_orb']]
	away_stats['drb'] += stats_schedule.loc[game,['away_drb']]
	away_stats['fgp'] += stats_schedule.loc[game,['away_fgp']]
	away_stats['ftp'] += stats_schedule.loc[game,['away_ftp']]
	away_stats['3pp'] += stats_schedule.loc[game,['away_3pp']]

	averages.loc[game_schedule.loc[game,'home_team'],'game_count'] += 1
	averages.loc[game_schedule.loc[game,'away_team'],'game_count'] += 1

	home_stats = home_stats / averages.loc[game_schedule.loc[game,'home_team'],'game_count']
	away_stats = away_stats / averages.loc[game_schedule.loc[game,'away_team'],'game_count']

	averages.loc[game_schedule.loc[game,'home_team'],averages_columns[1:]] = home_stats
	averages.loc[game_schedule.loc[game,'away_team'],averages_columns[1:]] = away_stats

print 'Total wins:',result[result['win_loss']==1].sum()['win_loss']
print 'Total losses:',-(result[result['win_loss']==-1].sum())['win_loss']

# print result
# print averages
*************************************************************************"""
