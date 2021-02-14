# area_classification

Main goal of this project is to make area classification for sentinel images

How to use it?
Get data from sentinel for examle from creodias.eu
Convert it to png files I used imagemagic and gimp
We have to classifi data for supervised learning and for evaluation process I used open strat map plugin for QGIS

config.ini
* input directory
* input images size
* output file
* classes names

preprocesing.py / creating_dataset.ipynb
* read images from Data folder
* read pixel classification from data classification folder
* remove outstanding data
* normalization
* saving to csv file
* + wisualization in jpynb version

heplers.py
* some usefoul functions for visualisation
* undersampling should be moved to preprocessing


Used liblaries: numpy, pandas, sklearn, opencv, minisom, matplotlib

