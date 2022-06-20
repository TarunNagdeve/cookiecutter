import pandas as pd
import numpy as np

df_train=pd.read_csv('/Users/trunnnnnn/Downloads/archive/train.csv')
df_test=pd.read_csv('/Users/trunnnnnn/Downloads/archive/test.csv')
df_train=df_train.dropna()

train_features=df_train[['x']]
train_labels=df_train[['y']]
test_features=df_test[['x']]
test_labels=df_test[['y']]

from sklearn.linear_model import LinearRegression
lmodel= LinearRegression()
lmodel.fit(train_features,train_labels)
pred=lmodel.predict(test_features)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(test_labels,pred)))
