# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg, nn_arch_madelon, nn_arch_digits, flag
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]

def read_integers(filename):
    with open(filename) as f:
        return [int(x) for x in f]

a = read_integers("./n_jobs.txt")
num_jobs = a[0]

out = './PCA/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')
madelonX = madelon.drop('Class',1).copy().values
madelonY = madelon['Class'].copy().values


madelonX = StandardScaler().fit_transform(madelonX)
digitsX= StandardScaler().fit_transform(digitsX)

#raise
#%% data for 1 just save explained_variance

pca = PCA(random_state=5)
pca.fit(madelonX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,pca.explained_variance_ratio_.shape[0]+1))
# print (pca.explained_variance_ratio_.shape)

tmp.to_csv(out+'madelon scree.csv')


pca = PCA(random_state=5)
pca.fit(digitsX)
tmp = pd.Series(data = pca.explained_variance_ratio_,index = range(1,pca.explained_variance_ratio_.shape[0]+1))
tmp.to_csv(out+'digits scree.csv')

# raise

#%% Data for 2, grid search on NN with PCA dimension reduction, NN alpha, and NN structure

if flag == 1:
    nn_arch=nn_arch_madelon
grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,n_jobs=num_jobs,verbose=10,cv=5)

gs.fit(madelonX,madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Madelon dim red.csv')

if flag == 1:
    nn_arch = nn_arch_digits
grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,n_jobs=num_jobs,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits dim red.csv')
# raise

#%% data for 3, Found proper number of dimension from previous, reduce dimension
# and save to datasets.hdf for clustering
# Set this from chart 2 and dump, use clustering script to finish up
dim = 5
pca = PCA(n_components=dim,random_state=10)

madelonX2 = pca.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

dim = 60
pca = PCA(n_components=dim,random_state=10)
digitsX2 = pca.fit_transform(digitsX)
digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
cols = list(range(digits2.shape[1]))
cols[-1] = 'Class'
digits2.columns = cols
digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)