import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv(r'', encoding='latin-1')
df.head()

df = df.drop(columns = ['PLAYER', 'CS'])
df.head()

X = df.iloc[:,0:12]
Y= df.iloc[:,12]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=11, test_size=0.2)
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)
print(y_pred)

knn.score(X_test , Y_test)

cm = confusion_matrix(Y_test, y_pred)
print(cm)

cr = classification_report(Y_test, y_pred)
print(cr)
