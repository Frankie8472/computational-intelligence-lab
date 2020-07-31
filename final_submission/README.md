# Project 1: Collaborative Filtering

**Computational Intelligence Lab 2020 (http://da.inf.ethz.ch/teaching/2020/CIL/)**

Authors: Josua Cantieni, Sarah Kamp, Franz Knobel, Silvia La

## Directory Structure

- data_interpretation.ipynb: Plot and interpret the data
- ./input
    - data_train.csv: the training set (https://www.kaggle.com/c/cil-collab-filtering-2020/data)
    - sampleSubmission.csv: a sample submission file in the correct format (https://www.kaggle.com/c/cil-collab-filtering-2020/data)
- ./baselines
    - ./als
        - **als.py**: computes the "Alternating Least Squares" algorithm on the data in '../../input/data_train.csv'
        - **als_data_util.py**: some helper functions for als.py
        - **README.md**: how to run the ALS baseline
        - **./output**: ALS output file is saved here as 'als_submission.csv'
    - ./svd
        - **run.py**: computes the "Singular Value Decomposition" algorithm on the data in '../../input/data_train.csv'
        - **fill_matrix.py**: some functions called by run.py to initially fill the prediction matrix
        - **improve.py**: some functions called by run.py to improve the predictions using SVD
        - **data_util.py**: some helper functions
        - **README.md**: how to run the SVD baseline
        - **./output**: SVD output file is saved here as 'svd_submission.csv'