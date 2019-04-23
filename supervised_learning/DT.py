#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:53:35 2018

@author: kenan
"""

'''suppress future warnings for ease of analysis'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
import random

'''load iris dataset split into train and test'''
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

'''load wine dataset split into train and test'''
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


'''set up the figure to have 2 subplots,iris and wine'''
f, (ax1,ax2) = plt.subplots(1,2, figsize= (15,6))
xaxis = [3,9,15,21]
labels = ['nFeatures test', 'nFeatures training',
          'sqrt Features test', 'sqrt Features training',
          'log2 Features test', 'log2 Features training']
plot_args = ['c', 'c--', 'k', 'k--','m', 'm--']

'''set up parameters for grid cross validation'''
params = [{'criterion': ['entropy'], 'max_features': [None, 'sqrt', 'log2'],
           'min_samples_split': [3], 'max_depth': [1,5,10,15,None]}]
    
DT = DecisionTreeClassifier()
irisCLF = GridSearchCV(DT, params)

'''fit iris'''
irisCLF.fit(irisX, irisY)
pred = irisCLF.predict(irisPred)
print('best estimator:', irisCLF.best_estimator_)
print('score:', metrics.accuracy_score(irisKnown, pred))
''' print results for analysis'''
print(pandas.DataFrame(irisCLF.cv_results_))
scores = irisCLF.cv_results_.get('mean_test_score')
trains = irisCLF.cv_results_.get('mean_train_score')
nFeatures= scores[0::3]
sqrtFeatures = scores[1::3]
log2Features = scores[2::3]
ntrain = trains[0::3]
sqrtTrain = trains[1::3]
log2Train = trains[2::3]
ind = params[0]['max_depth']

ax1.plot(ind,nFeatures, 'c')
ax1.plot(ind,ntrain, 'c--')
ax1.plot(ind,sqrtFeatures,'k')
ax1.plot(ind,sqrtTrain,'k--')
ax1.plot(ind,log2Features,'m')
ax1.plot(ind,log2Train,'m--')
ax1.set_title('Iris')
ax1.set_xlabel('Max Depth')
ax1.set_ylabel('Accuracy')

'''fit wine'''
wineCLF = GridSearchCV(DT, params)
wineCLF.fit(wineX, wineY)
pred = wineCLF.predict(winePred)
print('best estimator:', wineCLF.best_estimator_)
print('score:', metrics.accuracy_score(wineKnown, pred))
'''print results for analysis'''
print(pandas.DataFrame(wineCLF.cv_results_))
scores = wineCLF.cv_results_.get('mean_test_score')
trains = wineCLF.cv_results_.get('mean_train_score')
nFeatures = scores[0::3]
sqrtFeatures = scores[1::3]
log2features = scores[2::3]
ind = params[0]['max_depth']
ax2.plot(ind,nFeatures, 'c')
ax2.plot(ind,ntrain, 'c--')
ax2.plot(ind,sqrtFeatures,'k')
ax2.plot(ind,sqrtTrain,'k--')
ax2.plot(ind,log2Features,'m')
ax2.plot(ind,log2Train,'m--')
ax2.set_title('Iris')
ax2.set_xlabel('Max Depth')
ax2.set_ylabel('Accuracy')

f.legend(ax1.get_lines(), labels)
plt.show()

'''save figure as png for analysis'''
f.savefig('DTModelCom.png', dpi = 300)

params = {'criterion': ['entropy'], 'max_features': ['sqrt'], 'min_samples_split': [3], 'max_depth': [15]}
clf = GridSearchCV(DT, params)

scorings = []
training = []

p = [math.floor(3*irisLen/10),math.floor(4*irisLen/10),math.floor(5*irisLen/10),
     math.floor(6 * irisLen/ 10),math.floor(7*irisLen/10),math.floor(8*irisLen/10),
     math.floor(9*irisLen/10), irisLen]
for i in p:
    clf.fit(irisData[:i],irisTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))
    
xax = [30, 40,50,60,70,80,90,100]
f2, (ax3, ax4) = plt.subplots(1,2, figsize= (15,6))

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
     math.floor(6 * wineLen/ 10),math.floor(7*wineLen/10),math.floor(8*wineLen/10),
     math.floor(9*wineLen/10), wineLen]
for i in p:
    clf.fit(wineData[:i],wineTarget[0:i])
    scorings.append(clf.cv_results_.get('mean_test_score'))
    training.append(clf.cv_results_.get('mean_train_score'))
    
ax4.plot(xax, scorings, label='test', c='red', linestyle='-')
ax4.plot(xax,training, label='training',c='cyan', linestyle= '-')
ax4.set_title('Wine')
ax4.set_xlabel("Percent of Dataset")
ax4.set_ylabel("Accuracy")

f2.legend(ax3.get_lines(), labels)
'''saves the figure as a png'''
f2.savefig('learning_curveDT.png', dpi = 300)
plt.show()
