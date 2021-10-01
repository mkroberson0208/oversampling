# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:22:55 2021

@author: A5821
"""

import os
import numpy as np
import pandas as pd
import pyodbc
import statsmodels.api as sm
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score

os.chdir('Z:\\DATA\\11000\\11000\\VPSTAFF\\Asset Liability Management\\FRM Modeling\\Modeling\\CLOC\\2021 CLOC')

# Connect to Hadoop File System on On-Prem Data Lake
dsn = 'HadoopProd-Impala'
user = 'A5821'
pwd = 'N=vyF312'

connectstring = 'DSN={dsn};UID={un};PWD={pw}'.format(dsn=dsn, un=user, pw=pwd)
cnxn = pyodbc.connect(connectstring, autocommit=True)

query = """SELECT report_date,
prev_face_amt,
prev_max_loc,
prev_face_amt/prev_max_loc as util,
default_ind,
chargeoff,
chargeoff_bal,
account_age_m,
member_age,
age_at_orig
FROM frmmodelling_sec.cloc 
WHERE active_ind = 1 
AND Prod_name not like '%Business%'  
AND Year = 2017
"""

df = pd.read_sql(query,cnxn)
df.index = df['report_date']
df['total_bal'] = df.groupby(df.index)['chargeoff_bal'].sum()
df['w'] = df['chargeoff_bal']/df['total_bal']
df['default_ind'].sum()/df['default_ind'].count()

(df['default_ind']*df['chargeoff_bal']).sum()/df['chargeoff_bal'].sum()
df['chargeoff'].sum()/df['chargeoff_bal'].sum()

monthly_losses = df.groupby(df.index)['chargeoff'].sum()
monthly_def_rate = df.groupby(df.index)['default_ind'].sum()/df.groupby(df.index)['default_ind'].count()
monthly_loss_rate = df.groupby(df.index)['chargeoff'].sum()/df.groupby(df.index)['chargeoff_bal'].sum()
def annualize:
    return(1-(1))
fig, ax = plt.subplots(1)
ax.plot(monthly_def_rate)
ax.plot(monthly_loss_rate)


iv = ['member_age']
df_est = df[iv+['default_ind','w','chargeoff','chargeoff_bal']].dropna()
x = np.asarray(df_est[iv])
y = np.asarray(df_est['default_ind'])

d = {}
# Fit a logistic model
model = sm.GLM(endog=y, exog=sm.add_constant(x), family=sm.families.Binomial())
result = model.fit()
y_pred = (result.predict(sm.add_constant(x)) > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
r = {}
r['specificity'] = tn/(tn+fp)
r['sensitivity'] = tp/(tp+fn)
r['accuracy'] = (tp+tn)/len(y)
df_est['y_hat'] = result.predict(sm.add_constant(x))
df_est['chargeoff_pred'] = df_est['y_hat']*df_est['chargeoff_bal']
actuals = df_est.groupby(df_est.index)['chargeoff'].sum()
predictions = df_est.groupby(df_est.index)['chargeoff_pred'].sum()
r['MPE_chargeoffs'] = ((predictions - actuals)/actuals).mean()
r['MAPE_chargeoffs'] = (abs((predictions - actuals)/actuals)).mean()
r['pred_prob'] = df_est['y_hat'].mean()
r['observed_prob'] = y.mean()
r['avg_actuals'] = actuals.mean()
r['avg_prediction'] = predictions.mean()
d['baseline'] = r
print(r)

model = sm.GLM(endog=y, exog=sm.add_constant(x), family=sm.families.Binomial(),freq_weights=np.asarray(df_est['w']))
result = model.fit()
y_pred = (result.predict(sm.add_constant(x)) > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
r = {}
r['specificity'] = tn/(tn+fp)
r['sensitivity'] = tp/(tp+fn)
r['accuracy'] = (tp+tn)/len(y)
df_est['y_hat'] = result.predict(sm.add_constant(x))
df_est['chargeoff_pred'] = df_est['y_hat']*df_est['chargeoff_bal']
actuals = df_est.groupby(df_est.index)['chargeoff'].sum()
predictions = df_est.groupby(df_est.index)['chargeoff_pred'].sum()
r['MPE_chargeoffs'] = ((predictions - actuals)/actuals).mean()
r['MAPE_chargeoffs'] = (abs((predictions - actuals)/actuals)).mean()
r['pred_prob'] = df_est['y_hat'].mean()
r['observed_prob'] = y.mean()
r['avg_actuals'] = actuals.mean()
r['avg_prediction'] = predictions.mean()
d['bal-weighted'] = r
print(r)



for x in [0.5,0.33,0.25,0.2,0.1,0.05,0.025,0.01]:
    prop = int((1/x) - 1)
    sampling = str(int(x*100))+'% event rate'
    print(sampling)
    x_resampled, y_resampled = SMOTE(sampling_strategy=1/prop).fit_resample(x, y)
    model = sm.GLM(endog=y_resampled, exog=sm.add_constant(x_resampled), family=sm.families.Binomial())
    result = model.fit()
    y_pred = (result.predict(sm.add_constant(x)) > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    r = {}
    r['specificity'] = tn/(tn+fp)
    r['sensitivity'] = tp/(tp+fn)
    r['accuracy'] = (tp+tn)/len(y)
    df_est['y_hat'] = result.predict(sm.add_constant(x))
    df_est['chargeoff_pred'] = df_est['y_hat']*df_est['chargeoff_bal']
    actuals = df_est.groupby(df_est.index)['chargeoff'].sum()
    predictions = df_est.groupby(df_est.index)['chargeoff_pred'].sum()
    r['MPE_chargeoffs'] = ((predictions - actuals)/actuals).mean()
    r['MAPE_chargeoffs'] = (abs((predictions - actuals)/actuals)).mean()
    r['pred_prob'] = df_est['y_hat'].mean()
    r['observed_prob'] = y.mean()
    r['avg_actuals'] = actuals.mean()
    r['avg_prediction'] = predictions.mean()
    d[sampling] = r
    print(r)

    
results = pd.DataFrame.from_dict(d,orient='index')
os.chdir('Q:\\')
results.to_csv('oversampling.csv')
