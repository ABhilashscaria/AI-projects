import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import math
from matplotlib import style#
#style.use('ggplot')

df = pd.read_csv('GOOG.csv')
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 253)

df["HL_PCT"] = (df['High'] - df['Low']) / df['Low']
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open']
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
forecast_col = 'Close'
df.fillna(-9999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
pred = clf.predict(x_lately)
