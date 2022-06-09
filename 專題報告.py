# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:40:12 2022

@author: ivers
"""
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,linear_model
#載入資料
ins_original=pd.read_csv('insurance.csv')
ins=pd.read_csv('insurance.csv')
#檢查缺失值
ins.isna().sum()
#將類別資料轉換成數值資料
#sex; male=1;female=0
for i in range(len(ins['sex'])):
    if ins['sex'][i] == 'male':
        ins['sex'][i]=1
    else:
        ins['sex'][i]=0
#smoker; yes=1;no=0
for i in range(len(ins['smoker'])):
    if ins['smoker'][i] == 'yes':
        ins['smoker'][i]=1
    else:
        ins['smoker'][i]=0
#region; southwest=0;southeast=1;northwest=2;northeast=3
for i in range(len(ins['region'])):
    if ins['region'][i] == 'southwest':
        ins['region'][i]=0
    if ins['region'][i] == 'southeast':
        ins['region'][i]=1
    if ins['region'][i] == 'northwest':
        ins['region'][i]=2
    else:
        ins['region'][i]=3
        
####################################
#lm1
X1=pd.DataFrame(ins.iloc[:,0:6],columns=ins.iloc[:,0:6].columns)
y1=pd.DataFrame(ins.iloc[:,-1],columns=['charges'])
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=1234)

#lm2
X2=pd.DataFrame(ins.iloc[:,[0,2,3,4]],columns=ins.iloc[:,[0,2,3,4]].columns)
y2=pd.DataFrame(ins.iloc[:,-1],columns=['charges'])
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=1234)

from sklearn.linear_model import LinearRegression
#LinearRegression_1
lm1=LinearRegression()
lm1.fit(X_train1,y_train1)
r_square_lm_1=lm1.score(X_test1,y_test1)
mse_lm_1=np.mean( (y_test1-lm1.predict(X_test1))**2)[0]
mae_lm_1=np.mean(abs((y_test1-lm1.predict(X_test1))**2))[0]

df_lm1={'variable':['R Square:',"rmse:","mae:"],
        'LinearRegression_1':[r_square_lm_1,mse_lm_1**0.5,mae_lm_1]}
df_lm1=pd.DataFrame(df_lm1)

#LinearRegression_2
lm2=LinearRegression()
lm2.fit(X_train2,y_train2)
r_square_lm_2=lm2.score(X_test2,y_test2)
mse_lm_2=np.mean( (y_test2-lm2.predict(X_test2))**2)[0]
mae_lm_2=np.mean(abs((y_test2-lm2.predict(X_test2))**2))[0]

df_lm2={'variable':['R Square:',"rmse:","mae:"],
        'LinearRegression_2':[r_square_lm_2,mse_lm_2**0.5,mae_lm_2]}
df_lm2=pd.DataFrame(df_lm2)
lm1.intercept_
lm1.coef_
lm1_coef=[i for i in lm1.coef_[0]]
lm2.coef_
lm2_coef=[i for i in lm2.coef_[0]]


lm1_coef={'variable':['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
        'relative_importance_1':lm1_coef}
lm2_coef={'variable':['age', 'bmi', 'children', 'smoker'],
        'relative_importance_2':lm2_coef}

lm1_coef=pd.DataFrame(lm1_coef)
lm2_coef=pd.DataFrame(lm2_coef)
##############################################
import h2o
h2o.init()

#h2o_1
h2o_X1=[i for i in ins.columns[0:-1]]
h2o_y1=ins.columns[-1]
ins_1 = h2o.import_file("insurance.csv")
# Split the dataset into a train and valid set:
train1, valid1 = ins_1.split_frame(ratios=[.8], seed=14)

#h2o_2
h2o_X2=[i for i in ins.iloc[:,[0,2,3,4]].columns]
h2o_y2=ins.columns[-1]
# Import the cars dataset into H2O:
ins_2 = h2o.import_file("insurance.csv")
# Split the dataset into a train and valid set:
train2, valid2 = ins_2.split_frame(ratios=[.8], seed=14)

#H2ORandomForest_1
from h2o.estimators import H2ORandomForestEstimator
rf1 = H2ORandomForestEstimator(ntrees=10,nfolds = 5,seed=14)
rf1.train(x=h2o_X1,y=h2o_y1,training_frame=train1,validation_frame=valid1)
# Eval performance:
rf1.model_performance()
rf1
dfrf_1={'variable':['R Square:',"rmse:","mae:"],
        'H2O-RandomForest_1':[0.8298254,4927.001133530853,2843.236152455867]}
dfrf_1=pd.DataFrame(dfrf_1)

rf_1_={'variable':['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
        'percentage':[0.133051,0.006395,0.126829,0.022574,0.691056,0.020095]}
rf_1_=pd.DataFrame(rf_1_)

#H2ORandomForest_2
from h2o.estimators import H2ORandomForestEstimator
rf2 = H2ORandomForestEstimator(ntrees=10,nfolds = 5,seed=14)
rf2.train(x=h2o_X2,y=h2o_y2,training_frame=train2,validation_frame=valid2)
# Eval performance:
rf2.model_performance()
rf2

dfrf_2={'variable':['R Square:',"rmse:","mae:"],
        'H2O-RandomForest_2':[0.7958375,5410.189848608318,3487.9851510903286]}
dfrf_2=pd.DataFrame(dfrf_2)

rf_2_={'variable':['age', 'bmi', 'children', 'smoker'],
        'percentage':[0.142442,0.155679,0.026979,0.674900]}
rf_2_=pd.DataFrame(rf_2_)
####################################################
#H2OGradientBoosting_1
from h2o.estimators import H2OGradientBoostingEstimator
# Build and train the model:
gbm1 = H2OGradientBoostingEstimator(ntrees = 100,nfolds=5,seed=14)
gbm1.train(x=h2o_X1,y=h2o_y1,training_frame=train1,validation_frame=valid1)

# Eval performance:
gbm1.model_performance()
gbm1
# Extract feature interactions:
feature_interactions = gbm1.feature_interaction()

dfgbm_1={'variable':['R Square:',"rmse:","mae:"],
        'H2O-GBM_1':[0.851552,3378.2330759051383,1831.9521146307734]}
dfgbm_1=pd.DataFrame(dfgbm_1)

gbm_1={'variable':['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
        'percentage':[0.123738,0.002903,0.179014,0.012354,0.673337,0.008653]}
gbm_1=pd.DataFrame(gbm_1)

#H2OGradientBoosting_2
from h2o.estimators import H2OGradientBoostingEstimator
# Build and train the model:
gbm2 = H2OGradientBoostingEstimator(ntrees = 100,nfolds=5,seed=14)
gbm2.train(x=h2o_X2,y=h2o_y2,training_frame=train2,validation_frame=valid2)

# Eval performance:
gbm2.model_performance()
gbm2
# Extract feature interactions:
feature_interactions = gbm2.feature_interaction()

dfgbm_2={'variable':['R Square:',"rmse:","mae:"],
        'H2O-GBM_2':[0.8533988,3492.502644933548,1905.4744652244904]}
dfgbm_2=pd.DataFrame(dfgbm_2)

gbm_2={'variable':['age', 'bmi', 'children', 'smoker'],
        'percentage':[0.127283,0.183714,0.011648,0.677355]}
gbm_2=pd.DataFrame(gbm_2)

####################################################
con=sqlite3.connect('final.sqlite')
#建立資料表
sql_original='''CREATE TABLE IF NOT EXISTS original
("age" TEXT , "sex" TEXT, "bmi" TEXT, "children" TEXT, 
 "smoker" TEXT,"region" TEXT, "charges" TEXT) '''

sql_lm1_coef='''CREATE TABLE IF NOT EXISTS lm1_coef
("variable" TEXT, "relative_importance_1" TEXT)'''

sql_lm2_coef='''CREATE TABLE IF NOT EXISTS lm2_coef
("variable" TEXT, "relative_importance_2" TEXT)'''

sql_rf_1_='''CREATE TABLE IF NOT EXISTS rf_1_
("variable" TEXT, "percentage" TEXT)'''

sql_gbm_1='''CREATE TABLE IF NOT EXISTS gbm_1
("variable" TEXT, "percentage" TEXT)'''

sql_rf_2_='''CREATE TABLE IF NOT EXISTS rf_2_
("variable" TEXT, "percentage" TEXT)'''

sql_gbm_2='''CREATE TABLE IF NOT EXISTS gbm_2
("variable" TEXT, "percentage" TEXT)'''

sql_df_lm1='''CREATE TABLE IF NOT EXISTS df_lm1
("variable" TEXT,"LinearRegression_1" TEXT)'''

sql_df_lm2='''CREATE TABLE IF NOT EXISTS df_lm2
("variable" TEXT,"LinearRegression_2" TEXT)'''

sql_dfrf_1='''CREATE TABLE IF NOT EXISTS dfrf_1
("variable" TEXT,"RandomForest_1" TEXT)'''

sql_dfrf_2='''CREATE TABLE IF NOT EXISTS dfrf_2
("variable" TEXT,"RandomForest_2" TEXT)'''

sql_dfgbm_1='''CREATE TABLE IF NOT EXISTS dfgbm_1
("variable" TEXT,"GBM_1" TEXT)'''

sql_dfgbm_2='''CREATE TABLE IF NOT EXISTS dfgbm_2
("variable" TEXT,"GBM_2" TEXT)'''


con.execute(sql_original)
con.execute(sql_lm1_coef)
con.execute(sql_lm2_coef)
con.execute(sql_rf_1_)
con.execute(sql_gbm_1)
con.execute(sql_rf_2_)
con.execute(sql_gbm_2)
con.execute(sql_df_lm1)
con.execute(sql_df_lm2)
con.execute(sql_dfrf_1)
con.execute(sql_dfrf_2)
con.execute(sql_dfgbm_1)
con.execute(sql_dfgbm_2)

ins_original.to_sql('original',con,if_exists='replace',index=False)
ins.to_sql('test',con,if_exists='replace',index=False)
lm1_coef.to_sql('lm1_coef',con,if_exists='replace',index=False)
lm2_coef.to_sql('lm2_coef',con,if_exists='replace',index=False)
rf_1_.to_sql('rf_1_',con,if_exists='replace',index=False)
gbm_1.to_sql('gbm_1',con,if_exists='replace',index=False)
rf_2_.to_sql('rf_2_',con,if_exists='replace',index=False)
gbm_2.to_sql('gbm_2',con,if_exists='replace',index=False)
df_lm1.to_sql('df_lm1',con,if_exists='replace',index=False)
df_lm2.to_sql('df_lm2',con,if_exists='replace',index=False)
dfrf_1.to_sql('dfrf_1',con,if_exists='replace',index=False)
dfrf_2.to_sql('dfrf_2',con,if_exists='replace',index=False)
dfgbm_1.to_sql('dfgbm_1',con,if_exists='replace',index=False)
dfgbm_2.to_sql('dfgbm_2',con,if_exists='replace',index=False)

con.commit()
con.close()

