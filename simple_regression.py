import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, linear_model
import os

os.system('clear')

"""This method is purely for checking that the logic works"""
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# Read in csv table
df = pd.read_csv('nbascraper/command_results.csv')

df['home_fgp'] = df['home_fg'] / df['home_fga']
df['away_fgp'] = df['away_fg'] / df['away_fga']

df['home_3pp'] = df['home_3p'] / df['home_3pa']
df['away_3pp'] = df['away_3p'] / df['away_3pa']

df['home_ftp'] = df['home_ft'] / df['home_fta']
df['away_ftp'] = df['away_ft'] / df['away_fta']

df.drop(['home_fg','away_fg','home_fga','away_fga','home_3p','away_3p','home_3pa','away_3pa','home_ft','away_ft','home_fta','away_fta','home_ts','away_ts','home_efg','away_efg'],axis=1,inplace=True)

model_columns = df.drop(['home_team','away_team','game_date'],axis=1).columns

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

X = model_set.drop(['spread'],axis=1)#pd.DataFrame(columns = model_columns)

y = np.array(model_set.spread)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


clf = svm.SVR(kernel='linear')#linear_model.LogisticRegression(random_state=42)

clf = clf.fit(X_train,y_train)
print clf.score(X_test,y_test)


print clf.predict(X_test)
print y_test
