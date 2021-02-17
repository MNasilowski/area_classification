import cv2
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import MinMaxScaler


def images_to_numpy(input_dir, dx, dy, x=0, y=0):
    files = os.listdir(input_dir)
    print("diles: ", files)
    columns_names = []
    X = np.zeros((dx, dy, len(files)))
    for i, file in enumerate(files):
        file_path = os.path.join(input_dir, file)
        X[...,i] = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)[x:x+dx,y:y+dy]
        columns_names.append(file[-11:-8])
    data = X.reshape((dx*dy, len(files)))
    return data, columns_names

def remove_outstandings(data):
    for i in range(data.shape[1]):
        max_value = np.mean(data[:, i]) + 3*np.std(data[:, i])
        data[:, i] = (np.where(data[:, i] < max_value, data[:, i], max_value))
    sc = MinMaxScaler(feature_range=(0, 1))
    data = sc.fit_transform(data)
    return data

def get_classes(class_dir, dx, dy, x=0, y=0):
    
    files = os.listdir(class_dir)
    print("diles: ", files)
    X = np.zeros((dx, dy, len(files)))
    for i, file in enumerate(files):
        file_path = os.path.join(class_dir, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[x:x+dx,y:y+dy]
        X[...,i] = np.rint(img/255)
    classes = X.reshape((dx*dy, len(files)))
    return classes

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    input_dir = config['main']['input_dir']
    class_file = config['main']['classification_data']
    dx = int(config['main']['x_size'])
    dy = int(config['main']['y_size'])
    x = int(config['main']['x_start'])
    y = int(config['main']['y_start'])
    class_names = list(config['classes'].values())
    csv_data_file = config['main']['csv_data_file']

    data, columns_names = images_to_numpy(input_dir, dx, dy, x, y)
    data = remove_outstandings(data)
    classes = get_classes(class_file, dx, dy, x, y)
    other = (1 - classes.any(axis=1).astype(int)).reshape(-1,1)
    columns_names += class_names 
    data = np.concatenate((data, classes, other), axis=1)
    data = pd.DataFrame(data, columns=columns_names)
    data.to_csv(csv_data_file)

