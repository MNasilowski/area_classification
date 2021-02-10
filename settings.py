# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:48:27 2021

@author: nasil
"""

from PIL import Image 
x_size, y_size = Image.open("Data/Images/T34UDE_20200815T095039_B01.png").size

# I realy don't nead this file...
from configparser import ConfigParser
config = ConfigParser()
config.add_section('main')
config.set('main', 'x_size', str(x_size))
config.set('main', 'y_size', str(y_size))

config.add_section('classes')
config.set('classes', 'class_1', 'water')
config.set('classes', 'class_2', 'forest')
config.set('classes', 'class_3', 'fields')
config.set('classes', 'class_4', 'other')

with open('config.ini', 'w') as f:
    config.write(f)