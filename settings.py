# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:48:27 2021

@author: nasil
"""

from PIL import Image 







from configparser import ConfigParser
config = ConfigParser()

config.read('config.ini')
config.add_section('main')
config.set('main', 'key1', 'value1')
config.set('main', 'key2', 'value2')
config.set('main', 'key3', 'value3')


with open('config.ini', 'w') as f:
    config.write(f)