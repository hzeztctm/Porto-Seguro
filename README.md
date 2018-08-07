# Machine Learning Engineer Nanodegree 
# Capstone Project
## Project: Porto Seguro's Safe Driver Prediction

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [lightGBM](http://lightgbm.readthedocs.io/en/latest/index.html)

### Code
All ipython notebook are used for data preprocessing and feature engineering. 

### Data

All input data are in ./data folder and the detailed description of the data can be found in Kaggle. The dataset consists of over 600,000 data points, with each datapoint having 57 features. 
Download train.7z from:
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
Uncompress the file and put train.csv in /data/ 

**Features**
Porto Seguro labeled features that belong to similar groupings in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the post-fix bin and cat to represent binary and categorical features, respectively. Features without these designations are either continuous or ordinal. 

**Target Variable**
- `target`: 0 no claim; 1 claim

### Run

In a terminal or command window, navigate to the top-level project directory `PortoSeguro/` (that contains this README) and run one of the following commands:

```bash
ipython notebook PortoSeguro.ipynb
```  
or
```bash
jupyter notebook PortoSeguro.ipynb
```

These will open the iPython Notebook software and project file in your browser.

or
```bash
PortoSeguro.py
```

This will execute the script in Python.

### Documentation
See ./report/Report.pdf for detailed project documentation.

