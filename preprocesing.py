import cv2
import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import MinMaxScaler


def images_to_numpy(input_dir, dx, dy, x=0, y=0):
    files = os.listdir(input_dir)
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
    return data

def get_classes(class_dir, dx, dy, x=0, y=0):
    files = os.listdir(class_dir)
    columns_names = []
    X = np.zeros((dx, dy, len(files)))
    for i, file in enumerate(files):
        columns_names.append(file[:-4])
        file_path = os.path.join(class_dir, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)[x:x+dx,y:y+dy]
        X[...,i] = np.rint(img/255)
    classes = X.reshape((dx*dy, len(files)))
    return classes, columns_names

def add_classes_to_config(config, classes):
    config.remove_section('classes')
    config.add_section('classes')
    for i, name in enumerate(classes):
        config.set('classes', f'class_{i}', name)
    with open('config.ini', 'w') as f:
        config.write(f)
    return config 
    
if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    input_dir = config['main']['input_dir']
    class_file = config['main']['classification_data']
    dx = int(config['main']['x_size'])
    dy = int(config['main']['y_size'])
    x = int(config['main']['x_start'])
    y = int(config['main']['y_start'])
    csv_data_file = config['main']['csv_data_file']

    data, columns_names = images_to_numpy(input_dir, dx, dy, x, y)
    data = remove_outstandings(data)
    classes, class_names = get_classes(class_file, dx, dy, x, y)
    other = (1 - classes.any(axis=1).astype(int)).reshape(-1,1)
    class_names += ['other']
    add_classes_to_config(config, class_names)
    columns_names += class_names 
    data = np.concatenate((data, classes, other), axis=1)
    data = pd.DataFrame(data, columns=columns_names)
    data.to_csv(csv_data_file)
