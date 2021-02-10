# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:35:27 2021

@author: nasil
"""
import pandas as pd

def undersampling(df,classes_names,part=1):
    parts = {}
    for clas in classes_names:
        parts[clas] = df[df[clas]==1]
    min_len = [part.shape[0] for part in parts.values()]
    min_len = int(min(min_len)*part)
    for clas in classes_names:
        parts[clas] = parts[clas].sample(min_len)
    return pd.concat(parts.values())