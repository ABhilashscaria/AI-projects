import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
df = pd.read_csv('D:/python/Digit mnist/train.csv')

df = np.array(df)
m, n = df.shape
data = df[0:1000].T
Y_cv = data[0]
X_cv = data[1:n].T
X_cv = tf.keras.utils.normalize(X_cv)

data1 = df[1000:m].T
Y_train = data1[0]
X_train = data1[1:n].T
X_train = tf.keras.utils.normalize(X_train)


model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(10, input_shape=(784,), activation='relu'))
model.add(tf.keras.layers.Dense(10, input_shape=(10,), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs= 10 )
print(model.summary())

predictions = model.predict(X_cv)
pred = np.zeros((1000,1))
for i in range(0,1000):
    pred[i] = np.argmax(predictions[i])

print(accuracy_score(Y_cv, pred))
