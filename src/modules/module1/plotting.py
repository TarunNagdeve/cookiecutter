import pandas as pd
import numpy as np

df_train=pd.read_csv('/Users/trunnnnnn/Downloads/archive/train.csv')
import  matplotlib.pyplot as plt

plt.scatter(df_train[['x']],df_train['y'])
plt.show()