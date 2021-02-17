# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:35:27 2021

@author: nasil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def undersampling(df,classes_names,part=1):
    """
    dataframe undersampling. 
    Take random part of df so all classes will have the same number of samples
    df: pandas data frame with data
    classes_names list of df field that we will tread as classes
    part: float. You can take just part of the sample
    """
    parts = {}
    for clas in classes_names:
        parts[clas] = df[df[clas]==1]
    min_len = [part.shape[0] for part in parts.values()]
    min_len = int(min(min_len)*part)
    for clas in classes_names:
        parts[clas] = parts[clas].sample(min_len)
    return pd.concat(parts.values())

def IoU(target, predicted):
    """return intersection over union"""
    iou = np.sum(np.logical_and(target,predicted))
    iou = iou / (np.sum(np.logical_or(target,predicted)) + 1e-10)
    return iou

def metrics_matrix(Y_target, Y_pred, metric=IoU):
    """Compare target classes with predicted classes. 
    Return matrix with metrics
    """
    matrix = np.zeros((Y_pred.shape[1],Y_target.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):    
            matrix[i][j] = IoU(Y_target[:,j],Y_pred[:,i])
    return matrix

def show_target_pred_dif(yt,yp):
    """
    Show comparision beatween target and predicted image
    yt: target data 3D np.array 
    yp: predicted data 3D np.array 
    """
    column = ['Target','Predicted','Target - Predicted']
    nrows = yt.shape[-1]
    fig, axs = plt.subplots(figsize=(12,nrows*4), ncols = 3, nrows=nrows)
    [axi.set_axis_off() for axi in axs.ravel()]
    fig.tight_layout()
    for i in range(nrows):
        axs[i,0].imshow(yt[...,i]*255,cmap='gray', vmin=0, vmax=255)
        axs[i,1].imshow(yp[...,i]*255,cmap='gray', vmin=0, vmax=255)
        axs[i,2].imshow((yt[...,i]-yp[...,i]+1)*127,cmap='seismic', vmin=0, vmax=255)
    for ax, col in zip(axs[0], column):
        ax.set_title(col)
        
def plot_MinMaxAvg(data,figsize=(12,4)):
    """Plot mean values of data columns with min and max values"""
    average = data.mean(axis=0)
    st_deviation = data.std(axis=0)
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(np.arange(average.shape[0]), average, st_deviation, fmt='ok', lw=3)
    ax.errorbar(np.arange(average.shape[0]), average, [min_values,max_values], fmt='.k', ecolor='grey', lw=1)
    average.shape
    
def show_classes_distribution(data, classes):
    values = [np.sum(i) for i in data.T]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(classes,values)
    plt.show()
    
