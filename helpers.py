# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:35:27 2021

@author: nasil
"""
import pandas as pd
import numpy as np
import math
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

def acc(cm):
    """Return accuracy for class in confusion matrix"""
    accuracy = []
    for i in range(cm.shape[0]):
        accuracy.append((np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + 2*cm[i,i])/np.sum(cm))
    return accuracy

def prec(cm):
    """Return precisionfor class in confusion matrix"""
    precision = []
    for i in range(cm.shape[0]):
        precision.append(cm[i,i]/(np.sum(cm[:,i])))
    return precision

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
    fig, axs = plt.subplots(figsize=(10,nrows*3), ncols = 3, nrows=nrows)
    [axi.set_axis_off() for axi in axs.ravel()]
    fig.tight_layout()
    for i in range(nrows):
        axs[i,0].imshow(yt[...,i]*255,cmap='Reds', vmin=0, vmax=255)
        axs[i,1].imshow(yp[...,i]*255,cmap='Blues', vmin=0, vmax=255)
        axs[i,2].imshow((yt[...,i]-yp[...,i]+1)*127,cmap='seismic', vmin=0, vmax=255)
    for ax, col in zip(axs[0], column):
        ax.set_title(col)
       
        
def plot_MinMaxAvg(data, column_names, figsize=(12,4)):
    """Plot mean values of data columns with min and max values"""
    average = data.mean(axis=0)
    st_deviation = data.std(axis=0)
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    plt.figure()
    plt.errorbar(np.arange(average.shape[0]), average, st_deviation, fmt='ok',
                 lw=10, ecolor='dodgerblue')
    plt.errorbar(np.arange(average.shape[0]), average, st_deviation*3, fmt='ok',
                 lw=3, ecolor='dodgerblue')
    plt.errorbar(np.arange(average.shape[0]), average, [min_values,max_values],
                 fmt='.k', ecolor='grey', lw=1)
    plt.xticks(ticks=range(len(column_names)), labels=column_names)
    plt.xlabel("Bounds names")
    plt.ylabel("Value")
    plt.title('Pixsel values distribution in bounds: min, max, std')
    
def plot_values_histogram(bounds, column_names, ncols=3, nrows=2):
    """Plot bounds values histogram"""
    nrows=math.ceil(len(column_names)*1.0/ncols)
    fig, axs = plt.subplots(figsize=(3*ncols, 3*nrows),
                            ncols=ncols, nrows=nrows)
    fig.tight_layout(pad=3.0)
    for i, ax in enumerate(fig.get_axes()):
        if i >= len(column_names):
            break
        ax.hist(bounds[...,i].flatten(), bins=100)
        ax.set_title(column_names[i])
    plt.show()
    
def show_classes_distribution(data, classes):
    values = [np.sum(i) for i in data.T]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(classes,values)
    plt.show()
    
def plot_keras_history(history):
    fig, (ax1, ax2) = plt.subplots(figsize=(10,4), ncols=2, nrows=1)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set(xlabel='epoch', ylabel='accuracy')
    ax2.legend(['train', 'test'], loc='upper left')

    pass
    
