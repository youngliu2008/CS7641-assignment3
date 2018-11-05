import sys
import os
import time
from array import array
from time import clock
from itertools import product
import matplotlib.pyplot as plt
# from ggplot import *
import csv
import pandas as pd
import numpy as np
from helpers import  nn_arch,nn_reg
# from PCA import dims as PCA_dims
imgdir = './img/'
clusters = [2,5,10,15,20,25,30,35,40]
# clusters = [2,20, 40,60,80,100,120,240]
algnames = ['PCA','ICA','RP','SVD','RF']
clusternames = ['Kmeans','GMM']

datanames = ['Madelon', 'Digits']

def plot2D(dataname, algname):
    data = None
    fname = './@ALG@/@NAME@2D.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    fulldata = pd.read_csv(fname)

    plt.clf()

    imgfile = '@NAME@' + '_@ALG@' +"2D" + ".png"
    masks = []
    colors = ['b','g','r','c','m','y','k','w']
    for i,target in enumerate(fulldata['target'].unique()):

        mask = fulldata['target'] == target
        masks.append(mask)

        plt.scatter(fulldata['x'][mask], fulldata['y'][mask], c=colors[i], label = str(target))
        # plt.scatter(X[green, 0], X[green, 1], c="g")

        # plt.xlabel(x)
        # plt.ylabel(y)
        plt.legend()
        # plt.title(y + " vs " + x)
    # plt.show()
    plt.savefig(
        imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))

    return None
def plotTSNE(dataname, algname):
    data = None
    fname = './@ALG@/@NAME@2D.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    fulldata = pd.read_csv(fname)


    plt.clf()
    imgfile = '@NAME@' + '_@ALG@' +"TSNE" + ".png"
    masks = []
    colors = ['b','g','r','c','m','y','k','w']
    for i,target in enumerate(fulldata['target'].unique()):

        mask = fulldata['target'] == target
        masks.append(mask)

        plt.scatter(fulldata['x'][mask], fulldata['y'][mask], label=str(target), alpha = 0.5, s =10)
        # plt.scatter(fulldata['x'][mask], fulldata['y'][mask], c=colors[i%8], label = str(target))
        # plt.scatter(X[green, 0], X[green, 1], c="g")

        plt.xlabel('tSNE_x')
        plt.ylabel('tSNE_y')
        plt.legend()
        # plt.title(y + " vs " + x)
    # plt.show()
    plt.savefig(
        imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))


    # chart = ggplot(fulldata, aes(x='x', y='y', color='target')) \
    #         + geom_point(size=70, alpha=0.1) \
    #         + ggtitle("tSNE dimensions colored by Digit (PCA)")
    # chart.show()

    return None

def plotACC(dataname, algname):
    data = None
    plt.clf()
    imgfile = '@NAME@' +'_@ALG@' '_ACC' + '.png'


    fname = './@ALG@/@NAME@ acc.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    fulldata = pd.read_csv(fname)


    data = fulldata
    # print (data)
    # print (data.iloc[0,1:])


    plt.plot(clusters, data.iloc[0, 1:],label=algname+' GMM')
    plt.plot(clusters, data.iloc[1, 1:], label=algname+' Kmeans')

    plt.xlabel('cluster')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title( 'accuracy' + " vs " + 'cluster' )
    # plt.show()
    plt.savefig(imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))



    return None

def plotARI(dataname, algname):
    data = None
    plt.clf()
    imgfile = '@NAME@'+'_@ALG@' + '_ARI' + '.png'


    fname = './@ALG@/@NAME@ adjRI.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    fulldata = pd.read_csv(fname)


    data = fulldata
    # print (data)
    # print (data.iloc[0,1:])


    plt.plot(clusters,data.iloc[0,1:],label=algname+' GMM')
    plt.plot(clusters, data.iloc[1, 1:], label=algname+' Kmeans')

    plt.xlabel('cluster')
    plt.ylabel('Adjusted Rand Index')
    plt.legend()
    plt.title( 'Adjusted Rand Index' + " vs " + 'cluster' )
    # plt.show()
    plt.savefig(imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))

    return None

def plotScree(dataname, algname):
    data = None
    plt.clf()
    imgfile = '@NAME@'+'_@ALG@' + '_Scree' + '.png'

    if algname == 'RP':
        i = 1
        while i < 3:
            plt.clf()
            imgfile = '@NAME@_@ALG@_Scree' + str(i) + '.png'
            fname = './@ALG@/@NAME@ scree'.replace('@ALG@', algname).replace(
                '@NAME@', dataname) + str(i) + '.csv'
            data = pd.read_csv(fname, header = None)

            plt.plot(data.iloc[:, 0], data.iloc[:, 1])
            x= 'Components'
            if i == 1:
                y = 'Pairwise Distance Correlation'
            else:
                y = 'Reconstruction Error'
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(y + " vs " + x)
            # plt.legend()
            # plt.show()
            plt.savefig(
                imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                                   dataname))
            i += 1

    elif algname == 'PCA' or algname =='SVD':
        fname = './@ALG@/@NAME@ scree.csv'.replace('@ALG@', algname).replace(
            '@NAME@', dataname)
        data = pd.read_csv(fname, header = None)
        # print (data)
        print (data.iloc[:,1].size)
        cumsum = np.cumsum(data.iloc[:, 1])
        print (cumsum.size)

        plt.plot(data.iloc[:, 0], cumsum)

        x = 'Components'
        y = 'Cumulative Explained Variance'
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(y + " vs " + x)
        # plt.legend()
        # plt.show()
        plt.savefig(
            imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                               dataname))
    elif algname == 'ICA':
        fname = './@ALG@/@NAME@ scree.csv'.replace('@ALG@', algname).replace(
            '@NAME@', dataname)
        data = pd.read_csv(fname, header = None)
        # print (data)

        plt.plot(data.iloc[:, 0], data.iloc[:, 1])

        x = 'Components'
        y = 'Kurtosis'
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(y + " vs " + x)
        # plt.legend()
        # plt.show()
        plt.savefig(
            imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                               dataname))


    elif algname == 'RF':
        fname = './@ALG@/@NAME@ scree.csv'.replace('@ALG@', algname).replace(
            '@NAME@', dataname)
        data = pd.read_csv(fname, header=None)
        # print (data)
        plt.plot(data.iloc[:, 0], data.iloc[:, 1])

        x = 'Components'
        y = 'Feature Importance'
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(y + " vs " + x)
        # plt.legend()
        # plt.show()
        plt.savefig(
            imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                               dataname))
    else: raise

    return None

def plotLL(dataname, algname):
    data = None
    plt.clf()
    imgfile = '@NAME@'+'_@ALG@' + '_LogLike' + '.png'


    fname = './@ALG@/logliklihood.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    data = pd.read_csv(fname)

    if dataname == 'Digits':
        plt.plot(data.iloc[:, 0], data.iloc[:, 1])
    else:
        plt.plot(data.iloc[:, 0], data.iloc[:, 2])

    plt.xlabel('n_clusters')
    plt.ylabel('log likelihood')
    # plt.legend()
    plt.title('log likelihood' + " vs " + 'n_clusters')
    # plt.show()
    plt.savefig(
        imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                               dataname))

    return None

def plotSSE(dataname, algname):
    data = None
    plt.clf()
    imgfile = '@NAME@'+'_@ALG@' + '_SSE' + '.png'


    fname = './@ALG@/SSE.csv'.replace('@ALG@', algname).replace(
        '@NAME@', dataname)
    data = pd.read_csv(fname)

    if dataname == 'Digits':
        plt.plot(data.iloc[:, 0], data.iloc[:, 1])
    else:
        plt.plot(data.iloc[:, 0], data.iloc[:, 2])

    plt.xlabel('n_clusters')
    plt.ylabel('SSE')
    # plt.legend()
    plt.title('SSE' + " vs " + 'n_clusters')
    # plt.show()
    plt.savefig(
        imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@',
                                                               dataname))

    return None

def plot_dimred(dataname, algname, x, y):

    data = None
    fname = './@ALG@/@NAME@ dim red.csv'.replace('@ALG@',algname).replace('@NAME@',dataname)
    fulldata = pd.read_csv(fname)
    plt.clf()
    fig, ax = plt.subplots()
    imgfile = '@NAME@'+'_@ALG@' + ' dimred' + ".png"
    for nn_structure in nn_arch:
        for nn_alpha in nn_reg:
            data = fulldata.copy()
            data = data.loc[data['param_NN__hidden_layer_sizes'] == str(nn_structure)]
            data = data.loc[data['param_NN__alpha'] == nn_alpha]

            ax.plot(data[x], data[y],
                    label= \
                    # 'l='+
                    str(nn_structure)+\
                    # ' a=' +
                    str(nn_alpha)
                    )

    plt.xlabel(x)
    plt.ylabel(y)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title( y + " vs " + x )
    # plt.show()
    plt.savefig(imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))

    return None





def plot_cluster_NN(dataname, algname, clustername, x, y):

    fname = './@ALG@/@NAME@ cluster @clustername@.csv'\
        .replace('@ALG@',algname)\
        .replace('@NAME@',dataname)\
        .replace('@clustername@', clustername)
    fulldata = pd.read_csv(fname)
    plt.clf()

    imgfile = '@NAME@'+'_@ALG@ ' +clustername + '_clusterNN' + ".png"
    for nn_structure in nn_arch:
        for nn_alpha in nn_reg:
            data = fulldata.copy()
            data = data.loc[data['param_NN__hidden_layer_sizes'] == str(nn_structure)]
            data = data.loc[data['param_NN__alpha'] == nn_alpha]

            plt.plot(data[x], data[y],label='a='+str(nn_alpha)+' l='+str(nn_structure))
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.legend(loc = 3)
    plt.title( y + " vs " + x )
    # plt.show()
    plt.savefig(imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))

    return None

def label_bar(ax, bars, text_format, is_inside=True, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """
    max_y_value = max(bar.get_height() for bar in bars)
    if is_inside:
        distance = max_y_value * 0.05
    else:
        distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2
        if is_inside:
            text_y = bar.get_height() - distance
        else:
            text_y = bar.get_height() + distance

        ax.text(text_x, text_y, text, ha='center', va='bottom', **kwargs)

def plot_NN_Comparison(dataname, algnames, clusternames, value):
    categories = []
    km_numbers = []
    gmm_numbers=[]
    imgfile = dataname + ' NN '+value+' comp.png'
    plt.clf()
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.4

    categories =[]
    idx = 0
############# BASE Benchmark ##########
    fname = './BASE/@NAME@ NN bmk.csv'.replace('@NAME@', dataname)
    data = pd.read_csv(fname)
    base_nnStructure = 'BASE' \
                  +' '\
                  + data[data['rank_test_score'] == 1].iloc[0]['param_NN__hidden_layer_sizes'] \
                  # + '\n' \
                  # + ' alpha=' \
                  # + str(data[data['rank_test_score'] == 1].iloc[0]['param_NN__alpha'])
    # categories.append(nnStructure)
    mts = data.loc[1][value]
    # numbers.append(mts)

    benchmark = ax.bar(idx, mts, align='center', label='Benchmark',
                    width=bar_width, alpha=opacity, color='k', )
    categories.append(base_nnStructure + '-Benchmark')
    idx += 1
############## BASE dimension reduction only ########
    dim_idxs =[]
    number = []

    for algname in algnames:
        fname = './@ALG@/@NAME@ dim red.csv' \
            .replace('@ALG@', algname) \
            .replace('@NAME@', dataname) \

        data = pd.read_csv(fname)
        n_clusters_col= 'param_@ALG@__n_components'.replace('@ALG@', algname.lower())
        if algname == 'RF':
            n_clusters_col = 'param_filter__n'
        nnStructure = algname \
                      + ' '\
                      + data[data['rank_test_score'] == 1].iloc[0]['param_NN__hidden_layer_sizes'] \
                      + ' #cluster='+str(data[data['rank_test_score'] == 1].iloc[0][n_clusters_col])\
                      # + '\n' \
                      # + ' alpha=' + str(data[data['rank_test_score'] == 1].iloc[0]['param_NN__alpha'])

        categories.append(nnStructure)

        mts = data.loc[1][value]
        number.append(mts)
        dim_idxs.append(idx)
        idx += 1

    dimred_rects = ax.bar(dim_idxs, number, align='center', label='Dim Reduction',
                    width=bar_width, alpha=opacity, color='g', )
    # print (idx)
#############  BASE cluster only   ##########
    algname = 'BASE'

    km_idxs = []
    gmm_idxs =[]

    for i,clustername in enumerate(clusternames):
        fname = './@ALG@/@NAME@ cluster @clustername@.csv' \
            .replace('@ALG@', algname) \
            .replace('@NAME@', dataname) \
            .replace('@clustername@', clustername)
        data = pd.read_csv(fname)

        if clustername == 'Kmeans':
            n_clusters_col = 'param_union__km__n_clusters'
        else: n_clusters_col = 'param_union__gmm__n_components'
        # print (data[data['rank_test_score'] == 1].iloc[0][value])
        #
        # print (data[data['rank_test_score'] == 1].iloc[0]['param_NN__alpha'])

        nnStructure = algname \
                      + ' ' + clustername \
                      + ' ' + data[data['rank_test_score'] == 1].iloc[0]['param_NN__hidden_layer_sizes'] \
                      + ' #cluster=' + str(data[data['rank_test_score'] == 1].iloc[0][n_clusters_col]) \
            # + '\n' \
                      # + ' alpha=' + str(data[data['rank_test_score'] == 1].iloc[0]['param_NN__alpha'])
        # if i ==0:
        categories.append(nnStructure)
        mts = data.loc[1][value]
        if i ==0:
            km_numbers.append(mts)
            km_idxs.append(idx)
            idx += 1
        else:
            gmm_numbers.append(mts)
            gmm_idxs.append(idx)
            idx += 1


################ dimension reduction & cluster #########
    for i, clustername in enumerate(clusternames):
        for algname in algnames:
            fname = './@ALG@/@NAME@ cluster @clustername@.csv' \
                .replace('@ALG@', algname) \
                .replace('@NAME@', dataname) \
                .replace('@clustername@', clustername)
            data = pd.read_csv(fname)
            if clustername == 'Kmeans':
                n_clusters_col = 'param_union__km__n_clusters'
            else:
                n_clusters_col = 'param_union__gmm__n_components'
            nnStructure = algname \
                          + ' ' + clustername \
                          + ' ' \
                          + data[data['rank_test_score'] == 1].iloc[0]['param_NN__hidden_layer_sizes'] \
                          + ' #cluster=' + str(data[data['rank_test_score'] == 1].iloc[0][n_clusters_col]) \
                # + ' alpha=' + str(data[data['rank_test_score'] == 1].iloc[0]['param_NN__alpha'])

            # if i == 0:
            #     categories.append(nnStructure)
            categories.append(nnStructure)
            mts = data.loc[1][value]

            if i == 0:
                km_numbers.append(mts)
                km_idxs.append(idx)
                idx += 1
            else:
                gmm_numbers.append(mts)
                gmm_idxs.append(idx)
                idx += 1


    print (categories)
    km_rects = ax.bar(np.array(km_idxs), km_numbers,align='center',
                    width=bar_width, alpha=opacity, color='b', label = 'KM')
    gmm_rects = ax.bar(np.array(gmm_idxs), gmm_numbers, align='center',
                    width=bar_width, alpha=opacity, color='r', label = 'GMM')

    xticks_pos = np.arange(idx)
    print (xticks_pos)
    plt.xticks(xticks_pos, rotation=-90)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    if value in ['mean_fit_time','mean_score_time']:
        ax.legend(loc=0)
    else:
        ax.legend(loc=4)
    fig.tight_layout()
    value_format = "{:.2}"
    label_bar(ax,benchmark,value_format, is_inside=False, color="black")
    label_bar(ax, dimred_rects, value_format, is_inside=False, color="green")
    label_bar(ax, km_rects, value_format, is_inside=False, color="blue")
    label_bar(ax, gmm_rects, value_format, is_inside=False, color="red")
    # plt.show()
    plt.savefig(imgdir + imgfile.replace('@ALG@', algname).replace('@NAME@', dataname))

    return None


if __name__=='__main__':



    for dataname in datanames:
        for algname in algnames:
            plotScree(dataname=dataname,algname=algname)

    for dataname in datanames:
        for algname in ['BASE']+algnames:
            plotLL(dataname=dataname, algname=algname)

    for dataname in datanames:
        for algname in ['BASE']+algnames:
            plotSSE(dataname=dataname, algname=algname)
    for dataname in datanames:
        for algname in algnames:
            print('###########'+'dimred '+dataname+' '+algname)
            if algname == 'RF':
                x='param_filter__n'
            else:
                x='param_'+algname.lower()+'__n_components'
            y = 'mean_test_score'
            plot_dimred(dataname=dataname, algname=algname,
                        x=x, y=y)

    for dataname in datanames:
        for algname in algnames+['BASE']:
            print('###########' + 'TSNE ACC ARI ' + dataname + ' ' + algname)
            plotTSNE(dataname=dataname, algname=algname)
            plotACC(dataname=dataname, algname=algname)
            plotARI(dataname=dataname, algname=algname)

    for dataname in datanames:
        for algname in algnames:
            for clustername in clusternames:
                print('###########' + 'cluster ' + dataname + ' ' + algname+' '+clustername)
                # if clustername == 'Kmeans':
                #     x = 'param_km__n_clusters'
                # else: x ='param_gmm__n_components'
                if clustername == 'Kmeans':
                    x = 'param_union__km__n_clusters'
                else: x ='param_union__gmm__n_components'

                plot_cluster_NN(dataname=dataname,
                                algname=algname,
                                clustername=clustername,
                                x = x,
                                y = 'mean_test_score')
    values = ['mean_test_score','mean_fit_time','mean_score_time']
    for dataname in datanames:
        for value in values:
            print('###########' + 'comparison' + dataname )
            plot_NN_Comparison(dataname=dataname,algnames=algnames,clusternames=clusternames, value = value)