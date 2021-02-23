# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:48:27 2021

@author: nasil
"""

from configparser import ConfigParser
    
    
def r10m_part():
    config = ConfigParser()
    config.add_section('main')
    config.set('main', 'input_dir', 'Data/R10m_png')
    config.set('main', 'classification_data', 'Data/C10m')
    config.set('main', 'x_size', '2000')
    config.set('main', 'y_size', '2000')
    config.set('main', 'x_start', '3200')
    config.set('main', 'y_start', '240')
    config.set('main', 'csv_data_file', 'data_set.csv')
    config.add_section('classes')
    config.set('classes', 'class_1', 'water')
    config.set('classes', 'class_2', 'forest')
    config.set('classes', 'class_3', 'fields')
    config.set('classes', 'class_4', 'other')
    with open('config.ini', 'w') as f:
        config.write(f)


def r20m_part():
    config = ConfigParser()
    config.add_section('main')
    config.set('main', 'input_dir', 'Data/R20m_png')
    config.set('main', 'classification_data', 'Data/C20m')
    config.set('main', 'x_size', '1000')
    config.set('main', 'y_size', '1000')
    config.set('main', 'x_start', '1600')
    config.set('main', 'y_start', '120')
    config.set('main', 'csv_data_file', 'data_set.csv')
    config.add_section('classes')
    config.set('classes', 'class_1', 'water')
    config.set('classes', 'class_2', 'forest')
    config.set('classes', 'class_3', 'fields')
    config.set('classes', 'class_4', 'other')
    with open('config.ini', 'w') as f:
        config.write(f)    
        
        
def r60m_part():
    config = ConfigParser()
    config.add_section('main')
    config.set('main', 'input_dir', 'Data/R60m_png')
    config.set('main', 'classification_data', 'Data/C60m')
    config.set('main', 'x_size', '1000')
    config.set('main', 'y_size', '1000')
    config.set('main', 'x_start', '0')
    config.set('main', 'y_start', '0')
    config.set('main', 'csv_data_file', 'data_set.csv')
    config.add_section('classes')
    config.set('classes', 'class_1', 'water')
    config.set('classes', 'class_2', 'forest')
    config.set('classes', 'class_3', 'fields')
    config.set('classes', 'class_4', 'other')
    with open('config.ini', 'w') as f:
        config.write(f)            
        
        
        
if __name__ =='__main__':
    r10m_part()