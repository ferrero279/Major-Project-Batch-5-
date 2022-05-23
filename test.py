import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

dataset = pd.read_csv("Dataset/internet_scan_data.csv")
dataset.fillna(0, inplace = True)
dataset.drop(columns=['frame_info.encap_type', 'frame_info.time'],inplace=True)

cols = ['eth.type','ip.id','ip.flags','ip.checksum','ip.src','ip.dst','ip.dsfield','tcp.flags','tcp.checksum','label']

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
le7 = LabelEncoder()
le8 = LabelEncoder()
le9 = LabelEncoder()
le10 = LabelEncoder()

dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))
dataset[cols[3]] = pd.Series(le4.fit_transform(dataset[cols[3]].astype(str)))
dataset[cols[4]] = pd.Series(le5.fit_transform(dataset[cols[4]].astype(str)))
dataset[cols[5]] = pd.Series(le6.fit_transform(dataset[cols[5]].astype(str)))
dataset[cols[6]] = pd.Series(le7.fit_transform(dataset[cols[6]].astype(str)))
dataset[cols[7]] = pd.Series(le8.fit_transform(dataset[cols[7]].astype(str)))
dataset[cols[8]] = pd.Series(le9.fit_transform(dataset[cols[8]].astype(str)))
dataset[cols[9]] = pd.Series(le10.fit_transform(dataset[cols[9]].astype(str)))

dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

print(X)
print(Y)

X = X[0:20000]
Y = Y[0:20000]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = svm.SVC() 
cls.fit(X_train, y_train) 
predict = cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)









