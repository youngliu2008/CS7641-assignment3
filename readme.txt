CS 7641 Spring 2017 Assignment 3
(modified from JTay's original file)

This file describes the structure of this assignment submission. 
The assignment code is written in Python 3.6.6. Library dependencies are: 
scikit-learn 0.18.1
numpy 0.11.1
pandas 0.19.2
matplotlib 1.5.3

Other libraries used are part of the Python standard library. 

The main folder contains the following files:
1. readme.txt -> this file
2. yliu9-analysis.pdf -> The analysis for this assignment.
3. run.sh -> run this file to generate all data and plots. total process time ~48 hours on 24 core server with n_jobs = 20
4. helpers.py -> for constants, helper functions and classes 
5. parse.py -> to save unchanged Madelon and Digits datasets in ./BASE/datasets.hdf
6. benchmark.py -> to learn a NN on unchanged datasets, save results /BASE/
7. PCA.py -> to reduce dimension with PCA, save results in ./PCA/
8. ICA.py -> to reduce dimension with ICA, save results in ./ICA/
9. RP.py -> to reduce dimension with random projection, save results in ./RP/
10. RF.py -> to reduce dimension with random forest, save results in ./RF/
11. SVD.py -> to reduce dimension with SVD, sae results in ./SVD/
12. clustering.py -> to retrieve unchanged and dimension reduced dataset from each folder, run KMeans and GMM, then process with NN and save results in each directory
13. plot_a3.py -> to generate plots. I programmed this on a mac computer. It would not run on my ubuntu server because of case sensitivity in file names. 
14. ..madelon\ -> folder that include UCI madelon dataset
15. n_jobs.txt -> define number of jobs to run in parallel. should be smaller than number of cores.

To reproduce results, please first run run.sh . Each algorithm is also controlled by comment in the code. Please refer to the code comments.
