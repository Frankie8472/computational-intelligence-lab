# Collaborative Filtering

**[Computational Intelligence Lab 2020](http://da.inf.ethz.ch/teaching/2020/CIL/)**

Authors: Josua Cantieni, Sarah Kamp, Franz Knobel, Silvia La  
Group: Backpropagaters

This repository cointains all relevant code for the project. It contains the 
implementation of the the baseline algorithms and BCAF. Further, it contains the source files of the report.

## How to Run BCAF
1. Go into `src` folder.
```bash
cd src
```
2. Run EDP with SGD. The output is found in `./output/`
```bash
python3 edp_sgd_main.py
```
3. Run EDP with Adam. The output is found in `./output/`
```bash
python3 edp_adam_main.py
```
4. Do bootstrap aggregation. The output is found in `./output/bagging`
```bash
python3 bagging_main.py
```

## Dependencies
The run BCAF the following need to be installed:
- Python 3
- [Numpy](https://numpy.org/install/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [fastai](https://docs.fast.ai)

## Directory Structure

- `data_interpretation.ipynb`: Plot and interpret the data
- `./input`
    - `data_train.csv`: [training set](https://www.kaggle.com/c/cil-collab-filtering-2020/data)
    - `sampleSubmission.csv`: [sample submission file](https://www.kaggle.com/c/cil-collab-filtering-2020/data)
- `./baselines`
    - `./als`
        - `./output`: ALS output folder
	    - `README.md`: how to run the ALS baseline
        - `als.py`: computes the "Alternating Least Squares" algorithm on the training set
        - `als_data_util.py`: helper functions for als.py
    - `./svd`
        - `./output`: SVD output folder
        - `README.md`: how to run the SVD baseline
        - `run.py`: computes the "Singular Value Decomposition" algorithm on the training set
        - `fill_matrix.py`: helpfer functions to initially fill the prediction matrix
        - `improve.py`: helper functions to improve the predictions using SVD
        - `data_util.py`: some helper functions for data handling
    - `./user_similarities`
        - `./output`: KNN output folder
        - `README.md`: how to run the KNN baseline
        - `similarity.py`: computes the "k-nearest neighbors" algorithm on the training set
- `./src`
    - `bagging_main.py`: averages all csv submission files in `.output/bagging` and writes the result into a new csv file in `./output/bagging`
    - `edp_adam_main.py`: runs EDP with the Adam optimizer
    - `edp_sgd_data_util.py`: helper functions for edp_sgd_main.py
    - `edp_sgd_main.py`: runs EDP with the SGD optimizer in parallel for different values for 'k' and averages over all runs
- `./output`: output directory for the EDP algorithms in `./src`
    - `./bagging`: BACF output folder
- `./report`: all relevant files of the report
