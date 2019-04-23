#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:01:04 2018

@author: kenan
"""
'''suppress useless warnings for ease of analysis'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.svm import SVC
from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import pandas


'''load the iris dataset partition into training and testing'''
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

'''repeat for the wine dataset'''
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

'''set up figure to have 2 subplots, iris and wine'''
f, (ax1,ax2) = plt.subplots(1,2, figsize= (15,6))
xaxis = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]

labels = ['test', 'training']
plot_args = ['red', 'cyan']


'''set parameters for grid cross validation'''
params = [
        {'kernel': ['linear']},
        {'kernel' : ['poly'], 'degree' : [2, 3]},
        {'kernel': ['rbf'], 'gamma': [1], 'C': [.001, .01, .1, 1, 10, 100, 1000]}
        ]

svc = SVC()
clf = GridSearchCV(svc, params)
clf.fit(irisX, irisY)
pred = clf.predict(irisPred)

'''fit and print results for iris'''
print(pandas.DataFrame(clf.cv_results_))
print('best estimator:', clf.best_estimator_)
print('score:', metrics.accuracy_score(irisKnown, pred))
iscores = clf.cv_results_.get('mean_test_score')
itrains = clf.cv_results_.get('mean_train_score')

'''info from iris fitting is plotted on first plot'''
ax1.plot(xaxis,iscores[3:], label='test',
         c='red', linestyle= '-')
ax1.plot(xaxis,itrains[3:], label='training',
         c='cyan', linestyle= '-')
ax1.set_title('Iris')
ax1.set_xlabel("C")
ax1.set_ylabel("Accuracy")

'''fit and print results for wine'''
clf.fit(wineX, wineY)
pred = clf.predict(winePred)
print(pandas.DataFrame(clf.cv_results_))
print('best estimator:', clf.best_estimator_)
print('score:', metrics.accuracy_score(wineKnown, pred))
wscores = clf.cv_results_.get('mean_test_score')
wtrains = clf.cv_results_.get('mean_train_score')

'''plot to second subplot'''
ax2.plot(xaxis,iscores[3:], label='test',
         c='red', linestyle= '-')
ax2.plot(xaxis,itrains[3:], label='training',
         c='cyan', linestyle= '-')
ax2.set_title('Wine')
ax2.set_xlabel("C")
ax2.set_ylabel("Accuracy")

f.legend(ax1.get_lines(), labels)
plt.show()

'''save figure for analysis'''
f.savefig('ModComSVM.png', dpi = 300)


'''setup for Accuracy vs Percent of Dataset figure'''
xax = [30, 40,50,60,70,80,90,100]
f2, (ax3, ax4) = plt.subplots(1,2, figsize= (15,6))

params = {'kernel': ['rbf'], 'C': [10], 'gamma': [1]}
clf = GridSearchCV(svc, params)

'''Iris best solver'''
scorings = []
training = []
p = [math.floor(3*irisLen/10),math.floor(4*irisLen/10),math.floor(5*irisLen/10),
     math.floor(6 * irisLen / 10),math.floor(7*irisLen/10),math.floor(8*irisLen/10),
     math.floor(9*irisLen/10), irisLen]
for i in p:
    clf.fit(irisData[:i],irisTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))

ax3.plot(xax,scorings, label='test', c='red', linestyle= '-')
ax3.plot(xax,training, label='training', c='cyan', linestyle= '-')
ax3.set_title('Iris')
ax3.set_xlabel("Percent of Dataset")
ax3.set_ylabel("Accuracy")

'''Wine best solver'''
scorings = []
training = []
p = [math.floor(3*wineLen/10),math.floor(4*wineLen/10),math.floor(5*wineLen/10),
     math.floor(6 * wineLen / 10),math.floor(7*wineLen/10),math.floor(8*wineLen/10),
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


f2.legend(ax3.get_lines(), labels)

'''saves the figure as a png'''
f2.savefig('learning_curveSVM.png', dpi = 300)
plt.show()



