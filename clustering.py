# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

# %% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM, nn_arch, nn_reg, nn_arch_madelon, nn_arch_digits, flag
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin
import sys


def read_integers(filename):
    with open(filename) as f:
        return [int(x) for x in f]


a = read_integers("./n_jobs.txt")
num_jobs = a[0]

out = './{}/'.format(sys.argv[1])

np.random.seed(0)
digits = pd.read_hdf(out + 'datasets.hdf', 'digits')
digitsX = digits.drop('Class', 1).copy().values
digitsY = digits['Class'].copy().values

madelon = pd.read_hdf(out + 'datasets.hdf', 'madelon')
madelonX = madelon.drop('Class', 1).copy().values
madelonY = madelon['Class'].copy().values

madelonX = StandardScaler().fit_transform(madelonX)
digitsX = StandardScaler().fit_transform(digitsX)

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
# clusters = [2,20, 40,60,80,100,120,240]

# %% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
adjRI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)

    km.fit(madelonX)
    gmm.fit(madelonX)
    SSE[k]['Madelon'] = km.score(madelonX)
    ll[k]['Madelon'] = gmm.score(madelonX)
    acc[k]['Madelon']['Kmeans'] = cluster_acc(madelonY, km.predict(madelonX))
    acc[k]['Madelon']['GMM'] = cluster_acc(madelonY, gmm.predict(madelonX))
    adjMI[k]['Madelon']['Kmeans'] = ami(madelonY, km.predict(madelonX))
    adjMI[k]['Madelon']['GMM'] = ami(madelonY, gmm.predict(madelonX))
    adjRI[k]['Madelon']['Kmeans'] = ari(madelonY, km.predict(madelonX))
    adjRI[k]['Madelon']['GMM'] = ari(madelonY, gmm.predict(madelonX))

    km.fit(digitsX)
    gmm.fit(digitsX)
    SSE[k]['Digits'] = km.score(digitsX)
    ll[k]['Digits'] = gmm.score(digitsX)
    acc[k]['Digits']['Kmeans'] = cluster_acc(digitsY, km.predict(digitsX))
    acc[k]['Digits']['GMM'] = cluster_acc(digitsY, gmm.predict(digitsX))
    adjMI[k]['Digits']['Kmeans'] = ami(digitsY, km.predict(digitsX))
    adjMI[k]['Digits']['GMM'] = ami(digitsY, gmm.predict(digitsX))
    adjRI[k]['Digits']['Kmeans'] = ari(digitsY, km.predict(digitsX))
    adjRI[k]['Digits']['GMM'] = ari(digitsY, gmm.predict(digitsX))
    print(k, clock() - st)

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)
adjRI = pd.Panel(adjRI)

SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'logliklihood.csv')
acc.ix[:, :, 'Digits'].to_csv(out + 'Digits acc.csv')
acc.ix[:, :, 'Madelon'].to_csv(out + 'Madelon acc.csv')
adjMI.ix[:, :, 'Digits'].to_csv(out + 'Digits adjMI.csv')
adjMI.ix[:, :, 'Madelon'].to_csv(out + 'Madelon adjMI.csv')
adjRI.ix[:, :, 'Digits'].to_csv(out + 'Digits adjRI.csv')
adjRI.ix[:, :, 'Madelon'].to_csv(out + 'Madelon adjRI.csv')

# %% NN fit data (2,3)

class OrgFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer pass the original features
    """


    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

if flag == 1:
    nn_arch = nn_arch_madelon
grid = {'union__km__n_clusters': clusters, 'NN__alpha': nn_reg,
        'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True,
                    random_state=5)
km = kmeans(random_state=5)
# pipe = Pipeline([('km', km), ('NN', mlp)])

pipe = Pipeline([

    ('union', FeatureUnion(
        transformer_list=[
            ("OrgFeatures", OrgFeatures()),
            ('km', km)
        ])
     ),
    ('NN', mlp)
])
gs = GridSearchCV(pipe, grid, n_jobs=num_jobs, verbose=10)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon cluster Kmeans.csv')


if flag == 1:
    nn_arch = nn_arch_madelon
grid = {'union__gmm__n_components': clusters, 'NN__alpha': nn_reg,
        'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True,
                    random_state=5)
gmm = myGMM(random_state=5)
# pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
pipe = Pipeline([

    ('union', FeatureUnion(
        transformer_list=[
            ("OrgFeatures", OrgFeatures()),
            ('gmm', gmm)
        ])
     ),
    ('NN', mlp)
])
gs = GridSearchCV(pipe, grid, n_jobs=num_jobs, verbose=10, cv=5)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon cluster GMM.csv')



if flag == 1:
    nn_arch = nn_arch_digits
grid = {'union__km__n_clusters': clusters, 'NN__alpha': nn_reg,
        'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True,
                    random_state=5)
km = kmeans(random_state=5)
# pipe = Pipeline([('km', km), ('NN', mlp)])
pipe = Pipeline([

    ('union', FeatureUnion(
        transformer_list=[
            ("OrgFeatures", OrgFeatures()),
            ('km', km)
        ])
     ),
    ('NN', mlp)
])
gs = GridSearchCV(pipe, grid, n_jobs=num_jobs, verbose=10, cv=5)

gs.fit(digitsX, digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Digits cluster Kmeans.csv')



if flag == 1:
    nn_arch = nn_arch_digits
grid = {'union__gmm__n_components': clusters, 'NN__alpha': nn_reg,
        'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True,
                    random_state=5)
gmm = myGMM(random_state=5)
# pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
pipe = Pipeline([

    ('union', FeatureUnion(
        transformer_list=[
            ("OrgFeatures", OrgFeatures()),
            ('gmm', gmm)
        ])
     ),
    ('NN', mlp)
])
gs = GridSearchCV(pipe, grid, n_jobs=num_jobs, verbose=10, cv=5)

gs.fit(digitsX, digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Digits cluster GMM.csv')

# %% For chart 4/5
madelonX2D = TSNE(verbose=10, random_state=5).fit_transform(madelonX)
digitsX2D = TSNE(verbose=10, random_state=5).fit_transform(digitsX)

madelon2D = pd.DataFrame(np.hstack((madelonX2D, np.atleast_2d(madelonY).T)),
                         columns=['x', 'y', 'target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D, np.atleast_2d(digitsY).T)),
                        columns=['x', 'y', 'target'])

madelon2D.to_csv(out + 'madelon2D.csv')
digits2D.to_csv(out + 'digits2D.csv')