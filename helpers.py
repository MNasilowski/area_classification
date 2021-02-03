# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:35:27 2021

@author: nasil
"""

def undersampling(df,part=1):
# TODO Make it reusable
    forests = df[df["forest"]==1]
    water = df[df["water"]==1]
    fields = df[df["fields"]==1]
    other = df[df["other"]==1]
    min_len = min(forests.shape[0], water.shape[0], fields.shape[0], other.shape[0])
    min_len = int(min_len*part)
    forests = forests.sample(min_len)
    water = water.sample(min_len)
    fields = fields.sample(min_len)
    other = other.sample(min_len)
    return water.append([fields,forests, other])