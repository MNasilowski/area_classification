import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

files = os.listdir('Data')
dimension = 1830 # to powinno się dać jakoś automatycznie wczytać.
data = np.zeros((dimension, dimension, len(files)))
bands_names = []

for i, file in enumerate(files):
    bands_names.append(file[-7:-4])
    data[:, :, i] = cv2.imread(os.path.join('Data', file), cv2.IMREAD_UNCHANGED)[:dimension, :dimension]
X = data.reshape(dimension*dimension, len(files))

for i in range(X.shape[1]):
    X[:, i] = np.where(X[:, i] < np.mean(X[:, i]) + 3*np.std(X[:, i]), X[:, i], np.mean(X[:, i]) + 3*np.std(X[:, i]))
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

Y_real = cv2.imread('Classification.png')[:dimension, :dimension]
Y_real = Y_real.reshape((dimension*dimension, 3))
Y_real = (Y_real/255).astype(int)
print('aaa')
df = pd.DataFrame(np.concatenate((X, Y_real), axis=1), columns=bands_names + ['water', 'forest', 'fields'])
df.to_csv('data.csv')