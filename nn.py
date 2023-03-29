import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas import read_csv
import sklearn
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df = pd.read_excel('slecross.xlsx')
#print(df.head)

Y = df['Fraction'].to_numpy()

X = df.drop(['Fraction'], axis=1)
X = X.to_numpy()

train_cutoff = 850

trainX = X[:train_cutoff]
testX = X[train_cutoff:]

trainY = Y[:train_cutoff]
testY = Y[train_cutoff:]


model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
print(model.summary())
model.fit(trainX, trainY, validation_data=(testX, testY), verbose=0, epochs=100)

range_x = np.array([0.1, 0.25, 0.41, 0.6])
range_y = np.array([3, 5, 7, 9])
range_z = np.array([0.0005, 0.333, 0.667, 0.9993])

ycf_mean = 0.466  # input parameter x
stages_mean = 6  # input parameter y
rate_mean = 1515  # input parameter z

# z is fixed
mesh_z = np.meshgrid(range_x, range_y)
mesh_z = np.array(mesh_z)
mesh_z.shape

mesh_S = np.zeros((16, 3))
for i in range(4):
    for j in range(4):
        mesh_S[((4*i)+j), :] = np.array([rate_mean, mesh_z[:, 0, 2][0], mesh_z[:, 0, 2][0]])

mesh_S_prediction = model.predict(mesh_S)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(mesh_S[0], mesh_S[1], mesh_S_prediction, cmap=cm.coolwarm, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlim(0,1)
ax.set_ylim(0, 10)
plt.show()
