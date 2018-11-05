#!/usr/bin/env bash
python3.6 parse.py
python3.6 benchmark.py
python3.6 PCA.py
python3.6 ICA.py
python3.6 RP.py
python3.6 RF.py
python3.6 SVD.py
python3.6 clustering.py PCA
python3.6 clustering.py BASE
echo RP clustering
python3.6 clustering.py RP
echo RF clustering
python3.6 clustering.py RF
echo SVD clustering
python3.6 clustering.py SVD
echo ICA clustering
python3.6 clustering.py ICA
echo plotting
python3.6 plot_a3.py