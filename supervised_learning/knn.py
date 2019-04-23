#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:52:30 2018

@author: kenan
"""

'''suppress future warnings for ease of analysis'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import math
import random
import pandas
from sklearn.preprocessing import MinMaxScaler

'''load iris split into train and test'''
iris = datasets.load_iris()
irisData = MinMaxScaler().fit_transform(iris.data)
irisTarget = iris.target
irisLen = len(irisData)
irisShuffle = list(zip(irisData, irisTarget))
random.shuffle(irisShuffle)
irisData, irisTarget = zip(*irisShuffle)
irisX = irisData[:math.floor((3*irisLen)/5)]
irisY = irisTarget[:math.floor((3*irisLen)/5)]
irisPred = irisData[math.floor((3*irisLen)/5):]
irisKnown = irisTarget[math.floor((3*irisLen)/5):]

'''load wine split into train and test'''
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

'''set up the figure to have 2 subplots, iris and wine'''
f, (ax1,ax2) = plt.subplots(1,2, figsize= (15,6))
xaxis = [3,9,15,21]
labels = ['test', 'training']
plot_args = ['red', 'cyan']

'''set up the parameters'''
params = {'n_neighbors': [3,9,15,21]}

irisKNN = KNeighborsClassifier(weights='distance', algorithm= 'auto', n_jobs= -1)
irisCLF = GridSearchCV(irisKNN, params)
irisCLF.fit(irisX, irisY)
pred = irisCLF.predict(irisPred)
print('best estimator:', irisCLF.best_estimator_)
print('score:', metrics.accuracy_score(irisKnown, pred))

'''print the results'''
print(pandas.DataFrame(irisCLF.cv_results_))

iscores = irisCLF.cv_results_.get('mean_test_score')
itrains = irisCLF.cv_results_.get('mean_train_score')

'''plot iris fitting on first plot'''
ax1.plot(xaxis,iscores, label='test',
         c='red', linestyle= '-')
ax1.set_title('Iris')
ax1.set_xlabel("number of Nearest Neighbors")
ax1.set_ylabel("Accuracy")

wineKNN = KNeighborsClassifier(weights='distance', algorithm= 'auto', n_jobs= -1)
wineCLF = GridSearchCV(wineKNN, params)
wineCLF.fit(wineX, wineY)
pred = wineCLF.predict(winePred)

print('best estimator:', wineCLF.best_estimator_)
print('score:', metrics.accuracy_score(wineKnown, pred))

'''print the results'''
print(pandas.DataFrame(wineCLF.cv_results_))

wscores = wineCLF.cv_results_.get('mean_test_score')
wtrains = wineCLF.cv_results_.get('mean_train_score')

'''plot wine fitting on second plot'''
ax2.plot(xaxis,wscores, label='test',
         c='red', linestyle= '-')
ax2.set_title('Wine')
ax2.set_xlabel("number of Nearest Neighbors")
ax2.set_ylabel("Accuracy")

''' create legend, show plots, save figure'''
f.legend(ax1.get_lines(), labels)
plt.show()
f.savefig('knnModelCom.png', dpi = 300)

params = {'n_neighbors': [3]}
knn = KNeighborsClassifier(weights='distance', algorithm= 'auto', n_jobs= -1)
clf = GridSearchCV(knn, params)
scorings = []
training = []
xax = [30, 40,50,60,70,80,90,100]
f2, (ax3, ax4) = plt.subplots(1,2, figsize= (15,6))

p = [math.floor(3*irisLen/10),math.floor(4*irisLen/10),math.floor(5*irisLen/10),
     math.floor(6 * irisLen / 10),math.floor(7*irisLen/10),math.floor(8*irisLen/10),
     math.floor(9*irisLen/10), irisLen]
for i in p:
    clf.fit(irisData[:i],irisTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))
    
ax3.plot(xax,scorings, label='test',
         c='red', linestyle= '-')
ax3.plot(xax,training, label='training',
         c='cyan', linestyle= '-')
ax3.set_title('Iris')
ax3.set_xlabel("Percent of Dataset")
ax3.set_ylabel("Accuracy")

scorings = []
training = []

p = [math.floor(3*wineLen/10),math.floor(4*wineLen/10),math.floor(5*wineLen/10),
     math.floor(6 * wineLen / 10),math.floor(7*wineLen/10),math.floor(8*wineLen/10),
     math.floor(9*wineLen/10), wineLen]
for i in p:
    clf.fit(wineData[:i],wineTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))
    
ax4.plot(xax,scorings, label='test',
         c='red', linestyle= '-')
ax4.plot(xax,training, label='training',
         c='cyan', linestyle= '-')
ax4.set_title('Wine')
ax4.set_xlabel("Percent of Dataset")
ax4.set_ylabel("Accuracy")

f2.legend(ax3.get_lines(), labels)
'''saves the figure as a png'''
f2.savefig('learning_curveKNN.png', dpi = 300)
plt.show()


