#! env/bin/python3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import read_csv
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from prog import numerize_data

df = read_csv('data/Train_v2.csv')
df = df.drop(columns=['uniqueid'])

df = numerize_data(df)

df_ye = df[df['bank_account'] == 1]
df_no = df[df['bank_account'] == 0]

train_ye, test_ye = train_test_split(df_ye, test_size=0.3)
train_no, test_no = train_test_split(df_no, test_size=0.3)

train_ye = train_ye.sample(frac=1).reset_index(drop=True)
test_ye = test_ye.sample(frac=1).reset_index(drop=True)
train_no = train_no.sample(frac=1).reset_index(drop=True)
test_no = test_no.sample(frac=1).reset_index(drop=True)

train = pd.concat([train_ye, train_no])
train = train.sample(frac=1).reset_index(drop=True)
print(train['bank_account'])

test_ye = test_ye.drop(columns=['gender_of_respondent', 'cellphone_access', 'job_type', 'gender_of_respondent', 'relationship_with_head', 'marital_status'])
test_no = test_no.drop(columns=['gender_of_respondent', 'cellphone_access', 'job_type', 'gender_of_respondent', 'relationship_with_head', 'marital_status'])

#plt.bar(x, y)
#plt.show()


#print(frac)

y, X = dmatrices('bank_account ~ country + year + household_size + age_of_respondent + education_level', data=train)

#y, X = dmatrices('bank_account ~ country + year', data=df)
train_const = train.drop(columns=['bank_account', 'gender_of_respondent', 'job_type', 'cellphone_access', 'gender_of_respondent', 'relationship_with_head', 'marital_status'])

X = sm.add_constant(train_const)
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

print()

pred_ye = res.predict(test_ye)
pred_no = res.predict(test_no)

df_ye = pd.DataFrame(pred_ye)
df_no = pd.DataFrame(pred_no)

print(df_ye.describe())
print(df_no.describe())

