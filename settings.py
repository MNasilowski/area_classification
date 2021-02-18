# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:48:27 2021

@author: nasil
"""

from configparser import ConfigParser
    
    
def r10m_small():
    config = ConfigParser()
    config.add_section('main')
    config.set('main', 'input_dir', 'Data/R10m_png')
    config.set('main', 'classification_data', 'Data/C10m')
    config.set('main', 'x_size', '10')
    config.set('main', 'y_size', '10')
    config.set('main', 'x_start', '3200')
    config.set('main', 'y_start', '250')
    config.set('main', 'csv_data_file', 'data_set.csv')
    config.add_section('classes')
    config.set('classes', 'class_1', 'water')
    config.set('classes', 'class_2', 'forest')
    config.set('classes', 'class_3', 'fields')
    config.set('classes', 'class_4', 'other')
    with open('config.ini', 'w') as f:
        config.write(f)
    
r10m_small()