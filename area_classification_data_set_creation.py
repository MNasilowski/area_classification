import cv2
import os
import numpy as np
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
x_size = int(config['main']['x_size'])
y_size = int(config['main']['y_size'])
classes_names = list(config['classes'].values())

files = os.listdir('Data')
columns_names = []
x_size, y_size = 1830,1830
X = np.zeros((x_size,y_size,len(files)))
for i, file in enumerate(files):
    columns_names.append(file[-7:-4])
    X[...,i] = cv2.imread(os.path.join('Data', file), cv2.IMREAD_UNCHANGED)[:x_size,:y_size]
data = X.reshape((x_size*y_size, len(files)))

for i in range(data.shape[1]):
    max_value = np.mean(data[:, i]) + 3*np.std(data[:, i])
    data[:, i] = (np.where(data[:, i] < max_value, data[:, i], max_value))

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
data = sc.fit_transform(data)

classes = cv2.imread('data/classes/Classification.png')[:x_size, :y_size]
classes = classes.reshape((x_size*y_size, 3))
classes = (classes/255).astype(int)

other = (1 - classes.any(axis=1).astype(int)).reshape(-1,1)
columns_names += classes_names

data = np.concatenate((data, classes, other), axis=1)
data = pd.DataFrame(data, columns=columns_names)
data.to_csv('data.csv')