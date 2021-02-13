import cv2
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import MinMaxScaler


def images_to_numpy(input_dir,size):
    files = os.listdir(input_dir)
    columns_names = []
    X = np.zeros((size[0], size[1], len(files)))
    for i, file in enumerate(files):
        file_path = os.path.join(input_dir, file)
        X[...,i] = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        columns_names.append(file[-11:-8])
    data = X.reshape((size[0]*size[1], len(files)))
    return data, columns_names

def remove_outstandings(data):
    for i in range(data.shape[1]):
        max_value = np.mean(data[:, i]) + 3*np.std(data[:, i])
        data[:, i] = (np.where(data[:, i] < max_value, data[:, i], max_value))
    sc = MinMaxScaler(feature_range=(0, 1))
    data = sc.fit_transform(data)
    return data

def get_classes(class_dir, size):
    files = os.listdir(class_dir)
    X = np.zeros((size[0], size[1], len(files)))
    for i, file in enumerate(files):
        file_path = os.path.join(class_dir, file)
        X[...,i] = np.rint(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)/255)
    classes = X.reshape((size[0]*size[1], len(files)))
    return classes

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    input_dir = config['main']['input_dir']
    class_file = config['main']['classification_data']
    x_size = int(config['main']['x_size'])
    y_size = int(config['main']['y_size'])
    class_names = list(config['classes'].values())
    csv_data_file = config['main']['csv_data_file']

    data, columns_names = images_to_numpy(input_dir, (x_size,y_size))
    data = remove_outstandings(data)
    classes = get_classes(class_file, (x_size,y_size))
    other = (1 - classes.any(axis=1).astype(int)).reshape(-1,1)
    columns_names += class_names 
    data = np.concatenate((data, classes, other), axis=1)
    data = pd.DataFrame(data, columns=columns_names)
    data.to_csv(csv_data_file)
