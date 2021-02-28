# area_classification

Main goal of this project is to test some Machine Learning algorithms in area classification.
Available algorithms: SOM, Random Forest, Neural Network.
How to use it?
Get data from sentinel for example from creodias.eu
Convert it to png files I used imagemagic and gimp.
We need labelled data for supervised learning and for evaluation process. For this I used open strat map plugin for QGIS.
Prepeare config.ini
Run area_classification.jpynb or take a look at my results in area_classification.pdf

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
* some useful functions for visualisation
* undersampling

jp2topng.sh
* script for converting files jp2 from directory to png file in other directory

Used liblaries: numpy, pandas, sklearn, opencv, minisom, matplotlib

