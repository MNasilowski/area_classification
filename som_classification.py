import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import OneHotEncoder
from minisom import MiniSom
from helpers import undersampling
from configparser import ConfigParser

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
    


if  __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    x_size = int(config['main']['x_size'])
    y_size = int(config['main']['y_size'])
    class_names = list(config['classes'].values())
    csv_data_file = config['main']['csv_data_file']
    
    start_time = time.time() 
    df = pd.read_csv(csv_data_file)
    df = undersampling(df, class_names)
    X = df.iloc[:,1:14].to_numpy()
    Y_target = df.iloc[:,-4:].to_numpy()
    
    # Training Minisom
    som = MiniSom(x=6, y=6, input_len=13, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=200000, verbose=True)
    
    # Making prediction and encoding classes
    enc = OneHotEncoder()
    Y_pred = [som.winner(x) for x in X]
    Y_pred = np.array([y[0]*100 + y[1] for y in Y_pred ])
    Y_pred = enc.fit_transform(Y_pred.reshape(-1, 1)).toarray()
    
    # Evaluation
    metrix_IoU = metrics_matrix(Y_target, Y_pred, IoU)
    
    # Saving results
    pd.DataFrame(Y_pred).to_csv('Y_pred_from_som.csv')
    pickle.dump(som, open( "save.p", "wb" ) )
    print(f'Done in {time.time() - start_time}')



a = [1,2,3,4,5,6,7]
a[-4:]




