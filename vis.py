#! env/bin/python3

import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns

df = read_csv('data/Train_v2.csv')

#plt.bar(x, y)
#plt.show()


total = df[df['year'] == 2018]
frac = len( total[total['bank_account'] == 'Yes'] ) / len(total)

print(frac)
