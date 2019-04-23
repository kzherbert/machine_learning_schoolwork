#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:08:06 2018

@author: Kenan
"""

'''suppress useless warnings for ease of analysis'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV, ParameterGrid
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas
import math
import random

''' load the iris dataset then partition into training and testing'''
iris = datasets.load_iris()
irisData = MinMaxScaler(). fit_transform(iris.data)
irisTarget = iris.target
irisLen = len(irisData)
irisShuffle = list(zip(irisData, irisTarget))
random.shuffle(irisShuffle)
irisData, irisTarget = zip(*irisShuffle)
irisX = irisData[:math.floor((3*irisLen)/5)]
irisY = irisTarget[:math.floor((3*irisLen)/5)]
irisPred = irisData[math.floor((3*irisLen)/5):]
irisKnown = irisTarget[math.floor((3*irisLen)/5):]

''' laod the wine datatset then partition into training and testing'''
wine = datasets.load_wine()
wineData = MinMaxScaler().fit_transform(wine.data)
wineTarget = wine.target
wineLen = len(wineData)
wineShuffle = list(zip(wineData, wineTarget))
random.shuffle(wineShuffle)
wineData, wineTarget = zip(*wineShuffle)
wineX = wineData[:math.floor((3*wineLen)/5)]
wineY = wineTarget[:math.floor((3*wineLen)/5)]
winePred = wineData[math.floor((3*wineLen)/5):]
wineKnown = wineTarget[math.floor((3*wineLen)/5):]

''' Set up figure to have two subplots: iris and wine'''
f, (ax1, ax2) = plt.subplots(1,2, figsize= (15,6))
xaxis = [1,5,10,25,50,100,150,200,300,500,1500, 2000]
labels = ['testing(i)', 'testing(l)', 'testing(t)', 'testing(r)',
          'training(i)', 'training(l)', 'training(t)', 'training(r)']
plot_args = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
             'grey']

''' Set up parameters for grid cross validation'''
parameters = [
        {'activation': ['identity'], 'solver': ['adam'],
         'learning_rate_init': [0.01],
         'hidden_layer_sizes': [(1,100), (5,100), (10,100), (25,100), (50,100),
                                (100,100),(150,100), (200,100),(300,100),
                                (500, 400), (1500,400), (2000, 400)]},
        {'activation': ['logistic'], 'solver': ['adam'],
         'learning_rate_init': [0.01],
         'hidden_layer_sizes': [(1,100), (5,100), (10,100), (25,100), (50,100),
                                (100,100),(150,100), (200,100),(300,100), 
                                (500, 400), (1500,400), (2000, 400)]},
        {'activation': ['tanh'], 'solver': ['adam'],
         'learning_rate_init': [0.01],
         'hidden_layer_sizes': [(1,100), (5,100), (10,100), (25,100), (50,100),
                                (100,100),(150,100), (200,100),(300,100), 
                                (500, 400), (1500,400), (2000, 400)]},
        {'activation': ['relu'], 'solver': ['adam'], 
         'learning_rate_init': [0.01],
         'hidden_layer_sizes': [(1,100), (5,100), (10,100), (25,100), (50,100),
                                (100,100),(150,100), (200,100),(300,100),
                                (500, 400), (1500,400), (2000, 400)]},
        ]

''' Set up iris classifier'''
irisMLP = MLPClassifier(max_iter = 300)
irisCLF = GridSearchCV(irisMLP, parameters)
irisCLF.fit(irisX, irisY)
pred = irisCLF.predict(irisPred)

'''print dataframe'''
print(pandas.DataFrame(irisCLF.cv_results_))
print("best classifier: \n", irisCLF.best_estimator_)
print('score:', metrics.accuracy_score(irisKnown,pred))

iscores = irisCLF.cv_results_.get('mean_test_score')
itrains = irisCLF.cv_results_.get('mean_train_score')

'''info from iris fitting is plotted on first plot'''
ax1.plot(xaxis, iscores[0:12], label='testing(i)', c='blue', linestyle= '-')
ax1.plot(xaxis, iscores[12:24], label='testing(l)', c='green', linestyle= '-')
ax1.plot(xaxis, iscores[24:36], label='testing(t)', c='red', linestyle= '-')
ax1.plot(xaxis, iscores[36:48], label='testing(r)', c = 'cyan', linestyle= '-')
ax1.plot(xaxis, itrains[0:12], label='training(i)', c='magenta', linestyle= '-')
ax1.plot(xaxis, itrains[12:24], label='training(l)', c='yellow', linestyle= '-')
ax1.plot(xaxis, itrains[24:36], label='training(t)', c='black', linestyle= '-')
ax1.plot(xaxis, itrains[36:48], label='training(r)', color= '0.75', linestyle= '-')
ax1.set_title('Iris')
ax1.set_xlabel("Hidden Layer Sizes")
ax1.set_ylabel("Accuracy")

'''set up wine classifier'''
wineMLP = MLPClassifier(max_iter = 300)
wineCLF = GridSearchCV(wineMLP, parameters)
wineCLF.fit(wineX, wineY)
pred = wineCLF.predict(winePred)

'''print dataframe again'''
print(pandas.DataFrame(wineCLF.cv_results_))
print("best classifier: \n", wineCLF.best_estimator_)
print('score:', metrics.accuracy_score(wineKnown,pred))

wscores = wineCLF.cv_results_.get('mean_test_score')
wtrains = wineCLF.cv_results_.get('mean_train_score')

'''info from wine fitting plotted on second plot'''
ax2.plot(xaxis, wscores[0:12], label='testing(i)', c='blue', linestyle= '-')
ax2.plot(xaxis, wscores[12:24], label='testing(l)', c='green', linestyle= '-')
ax2.plot(xaxis, wscores[24:36], label='testing(t)', c='red', linestyle= '-')
ax2.plot(xaxis, wscores[36:48], label='testing(r)', c = 'cyan', linestyle= '-')
ax2.plot(xaxis, wtrains[0:12], label='training(i)', c='magenta', linestyle= '-')
ax2.plot(xaxis, wtrains[12:24], label='training(l)', c='yellow', linestyle= '-')
ax2.plot(xaxis, wtrains[24:36], label='training(t)', c='black', linestyle= '-')
ax2.plot(xaxis, wtrains[36:48], label='training(r)', color= '0.75', linestyle= '-')
ax2.set_title('Wine')
ax2.set_xlabel("Hidden Layer Sizes")
ax2.set_ylabel("Accuracy")

f.legend(ax1.get_lines(), labels)
plt.show()

'''fig is saved for analysis'''
f.savefig('neuralnetnn.png', dpi = 300)

'''Run the best activation function on splices of the dataset to find the learning curve'''
params = {'activation': ['relu'], 'solver': ['adam'], 'learning_rate_init': [0.01]}

MLP = MLPClassifier(max_iter=300)
clf = GridSearchCV(MLP, params)
scorings = []
training = []
p = [math.floor(3*irisLen/10),math.floor(4*irisLen/10),math.floor(5*irisLen/10),
     math.floor(6 * irisLen/10),math.floor(7*irisLen/10),math.floor(8*irisLen/10),
     math.floor(9*irisLen/10), irisLen]
for i in p:
    clf.fit(irisData[:i],irisTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))
    
xax = [30, 40,50,60,70,80,90,100]
f2, (ax3, ax4) = plt.subplots(1,2, figsize= (15,6))

ax3.plot(xax,scorings, label='test', c='red', linestyle= '-')
ax3.plot(xax,training, label='training', c='cyan', linestyle= '-')
ax3.set_title('Iris')
ax3.set_xlabel("Percent of Dataset")
ax3.set_ylabel("Accuracy")

'''do again for wine dataset'''
MLP = MLPClassifier(max_iter=300)
clf = GridSearchCV(MLP, params)
scorings = []
training = []
p = [math.floor(3*wineLen/10),math.floor(4*wineLen/10),math.floor(5*wineLen/10),
     math.floor(6 * wineLen/10),math.floor(7*wineLen/10),math.floor(8*wineLen/10),
     math.floor(9*wineLen/10), wineLen]
for i in p:
    clf.fit(wineData[:i],wineTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))

ax4.plot(xax,scorings, label='test', c='red', linestyle= '-')
ax4.plot(xax,training, label='training', c='cyan', linestyle= '-')
ax4.set_title('Wine')
ax4.set_xlabel("Percent of Dataset")
ax4.set_ylabel("Accuracy")

f2.legend(ax3.get_lines(), ['testing', 'training'])
f2.savefig('learning_curvenn.png', dpi = 300)
plt.show()




