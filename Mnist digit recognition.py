import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
df = pd.read_csv('D:/python/Digit mnist/train.csv')

df = np.array(df)
m, n = df.shape
data = df[0:1000].T
X = df[:, 1:m]
X = tf.keras.utils.normalize(X)
Y = df[:, 0]
Y = Y.reshape(42000, 1)
X_train = X[1000:m]
Y_train = Y[1000:m]
print(X_train.shape)
print(Y_train.shape)
X_test = X[0:1000]
Y_test = Y[0:1000]


model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape = (784,)))
model.add(tf.keras.layers.Dense(600, activation='relu')) #hidden layer
model.add(tf.keras.layers.Dense(64, activation='relu'))  #hidden layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs= 10 )
print(model.summary())

predictions = model.predict(X_test)
pred = np.zeros((1000,1))
for i in range(0,1000):
    pred[i] = np.argmax(predictions[i])

print(accuracy_score(Y_test, pred))
