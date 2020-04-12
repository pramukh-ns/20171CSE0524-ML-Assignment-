import numpy as np 
import pandas as pd 
df=pd.read_csv("Net.csv")
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
X=df[['show_id']]
Y=df[['release_year']]
X_train,X_test,y_train,y_test= train_test_split(X, Y, test_size = 0.2, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=7) 
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
p1=knn.predict([['80106440']])
p1[0]
