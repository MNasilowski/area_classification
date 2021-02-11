# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:48:27 2021

@author: nasil
"""

from PIL import Image
import os

def get_size(input_dir):
    """ Check resolution of image in input dir
    ToDo: assert that all images have the same resolution
    """
    file = os.listdir(input_dir)[0]
    x, y = Image.open(os.path.join(input_dir,file)).size[:2]
    return x, y

from configparser import ConfigParser
config = ConfigParser()
config.add_section('main')
config.set('main', 'input_dir', 'Data/R60m_png')
config.set('main', 'classification_data', 'Data/Classification.png')
x_size, y_size = get_size(config['main']['input_dir'])
config.set('main', 'x_size', str(x_size))
config.set('main', 'y_size', str(y_size))
config.set('main', 'csv_data_file', 'data_set.csv')
config.add_section('classes')
config.set('classes', 'class_1', 'water')
config.set('classes', 'class_2', 'forest')
config.set('classes', 'class_3', 'fields')
config.set('classes', 'class_4', 'other')

with open('config.ini', 'w') as f:
    config.write(f)
    
    
