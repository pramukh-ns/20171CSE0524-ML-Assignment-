import pandas as pd  
import numpy as np  
df=pd.read_csv("Net.csv")

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
X=df['show_id'].values.reshape(-1,1)
Y=df['release_year']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
p1 = regressor.predict([[80119451]])
p1[0]
