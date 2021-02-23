#!/usr/bin/env python
# coding: utf-8

# # Automatyczna klasyfikacja terenu z wykorzystaniem uczenia maszynowego

# Zrodlo danych: Sentinel T34UDE_20200815T095039
# Kroki wykonane przed analizą:
# * Zmiana formatu warstw z jp2 na png ze względu na bezproblemową współpracę z OpenCV
# * Klasyfikacja terenu z QGIS oraz wtyczkę QuickOSM
#  ** water
#  ** forest
#  ** farmland
# * Dla każdego typu terenu utworzono maskę w formie obrazu PNG o rozdzielczości zgodnej z danymi wejściowymi
# * Utworzenie pliku konfiguracyjnego config.ini

# #### Wczytanie bibliotek

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from configparser import ConfigParser


# In[2]:


import preprocesing as pre
import helpers as hlp


# #### Wczytanie danyc z pliku konfiguracyjnego

# In[3]:


config = ConfigParser()
config.read('config.ini')
input_dir = config['main']['input_dir']# Folder ze zdjęciami z Sentinela
classes_file = config['main']['classification_data']# Folder z maskami klas


# Zdjęcia o rozdzielczości 10m składają sie z ponad 100 milionów pikseli
# zatem do analizy wykorzystam tylko jego fragment o rozmiarze dx na dy
# i zaczynający się od piksela (x_star, y_start)

# In[4]:


dx = int(config['main']['x_size'])
dy = int(config['main']['y_size'])
x_start = int(config['main']['x_start'])
y_start = int(config['main']['y_start'])
csv_data_file = config['main']['csv_data_file']


# #### Przekształcamy dane wejściowe w coś przyjemniejszego do analizy

# In[5]:


data, columns_names = pre.images_to_numpy(input_dir, dx, dy, x_start, y_start)


# In[6]:


hlp.plot_MinMaxAvg(data, columns_names)


# In[7]:


hlp.plot_values_histogram(data, columns_names, ncols=5)


# Rozkład wartości pikseli wskazuje na występowanie wartości odstających zatem przekształćmy je w następujący sposób: $ x = min(x,\overline{x}+3\sigma_{x}) $ oraz przeskalujmy z wykorzystaniem minmaxscaler z sklearn.

# In[8]:


data = pre.remove_outstandings(data)


# In[9]:


hlp.plot_MinMaxAvg(data, columns_names)


# In[10]:


hlp.plot_values_histogram(data, columns_names, ncols=5)


# #### Wczytajmy teraz maski klas oraz stwórzmy klasę "other"

# In[11]:


classes, classes_names = pre.get_classes(classes_file, dx, dy, x_start, y_start)
other = (1 - classes.any(axis=1).astype(int)).reshape(-1, 1)
classes_names += ['other']
pre.add_classes_to_config(config, classes_names)
columns_names += classes_names 
nr_of_classes = len(classes_names)


# In[12]:


hlp.show_classes_distribution(np.concatenate((classes, other), axis=1), classes_names)


# In[13]:


data = np.concatenate((data, classes, other), axis=1)
data = pd.DataFrame(data, columns=columns_names)
data[classes_names] = data[classes_names].astype('int')


# In[14]:


data.head()


# Dane zostały przygotowane zapisujemy je i możemy zająć się klasyfikacją

# In[15]:


data.to_csv(csv_data_file)


# #### Dzielimy dane

# In[16]:


X = data.iloc[:,1:-nr_of_classes].to_numpy()
Y = data.iloc[:,-nr_of_classes:].to_numpy()
Y_2D = Y.reshape((dx,dy,nr_of_classes)) 


# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# # Klasyfikacja obszaru z wykorzystaniem lasów losowych

# #### Trenowanie

# In[18]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)


# #### Testowanie

# In[19]:


Y_pred_RF = clf.predict(X_test)


# In[20]:


print("Random forest acc: ",accuracy_score(Y_test, Y_pred_RF))


# #### Wizualizacja wyników

# In[21]:


Y_pred_RF = clf.predict(X)
Y_pred_RF = np.rint(Y_pred_RF)
hlp.show_target_pred_dif(Y_2D, Y_pred_RF.reshape((dx, dy, nr_of_classes)))


# # Klasyfikacja obszaru z wykorzystaniem sieci neuronowych

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[23]:


model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])


# In[24]:


model.summary()


# In[25]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[26]:


model.fit(X_train, Y_train, epochs=20, batch_size=10000)


# In[27]:


_, accuracy = model.evaluate(X_test, Y_test)
print(f'Deap learning acc: {accuracy}')


# #### Wizualizacja wyników

# In[28]:


Y_pred_DL = model.predict(X)
Y_pred_DL = np.rint(Y_pred_DL)
hlp.show_target_pred_dif(Y_2D,Y_pred_RF.reshape((dx, dy, nr_of_classes)))


# # Klasyfikacja obszaru z wykorzystaniem samoorganizujących się map

# In[29]:


from minisom import MiniSom
x_som, y_som = 5,5
som = MiniSom(x=x_som, y=y_som, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=10000, verbose=False)


# #### Klasyfikujemy
# Każdemu punktowi możemy przypisać jeden z neuronów mapy

# In[30]:


Y_pred_SOM = [som.winner(x) for x in X]
Y_pred_SOM = np.array([i[0]*100 + i[1] for i in Y_pred_SOM ])


# #### One Hot Encode

# In[31]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
Y_pred_SOM = enc.fit_transform(Y_pred_SOM.reshape(-1, 1)).toarray()


# In[32]:


hlp.show_classes_distribution(Y_pred_SOM, list(range(25)))


# In[33]:


mapa = np.sum(Y_pred_SOM*som.distance_map().flatten(), axis=1)


# #### rysujemy mapę

# In[34]:


fig, (ax1, ax2) = plt.subplots(figsize=(10,4), ncols=2, nrows=1)
ax1.set_title('SOM')
ax1.imshow(som.distance_map())
for (i, j), z in np.ndenumerate(som.distance_map()):
    ax1.text(j, i, '{:0.2f}'.format(som.distance_map()[i,j]), ha='center', va='center',color = 'white')
ax2.set_title('After classification')
ax2.imshow(mapa.reshape((dx,dy)))


# #### Walidacja

# In[35]:


clusstered = np.zeros((x_som,y_som,3))
matrix_IoU = hlp.metrics_matrix(Y, Y_pred_SOM, hlp.IoU)
clusstered[...,2]=matrix_IoU[:,0].reshape((x_som,y_som))#Klasa 1 kolor niebieski
clusstered[...,1]=matrix_IoU[:,1].reshape((x_som,y_som))#Klasa 2 kolor zielony
clusstered[...,0]=matrix_IoU[:,2].reshape((x_som,y_som))#Klasa 3 kolor czerwony
fig, ax = plt.subplots()
ax.set_title('Intersection over union')
plt.imshow(clusstered)
for (i, j, k), z in np.ndenumerate(clusstered):
    if z > 0.05:
        ax.text(j, i, '{:0.2f}'.format(max(clusstered[i,j,:])), ha='center', va='center',color = 'white')


# Kolorami oznaczono klasę którą reprezentują. Słabe wyniki spowodowane są dużo wyższą liczbą otrzymanych klas niż klas które mieliśmy początkowo.

# In[36]:


best = [matrix_IoU[:,i].argsort()[-1] for i in range(3)]


# In[37]:


hlp.show_target_pred_dif(Y_2D[...,:-1], Y_pred_SOM.reshape((dx,dy,Y_pred_SOM.shape[1]))[...,best])


# # Porównanie skuteczności algorytmów

# In[38]:


Y_RF = clf.predict(X)


# In[39]:


hlp.show_classes_distribution(Y_RF, classes_names)


# In[40]:


Y_DL = model.predict(X)


# In[41]:


Y_DL = np.rint(Y_DL)


# In[42]:


hlp.show_classes_distribution(Y_DL, classes_names)


# In[43]:


hlp.show_target_pred_dif(Y_RF.reshape((dx,dy,nr_of_classes)),Y_DL.reshape((dx,dy,nr_of_classes)))


# In[ ]:





# # Analiza wyników

# In[ ]:





# In[ ]:




