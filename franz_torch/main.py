from scipy.signal.filter_design import EPSILON
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from fastai.collab import *
import fastai
import torch
import os


class Parameters:
    def __init__(self):
        # User defined parameters
        self.REG = True
        self.WORKING_PATH = "../franz_torch"
        self.DATA_SET_PATH = "../input/data_train.csv"
        self.RES_SET_PATH = "../output/sampleSubmission.csv"
        self.SAVE_MODEL_PATH = "./model.pth"
        self.DTYPE = torch.double
        self.SEED = 42
        self.WORKERS = 3
        self.PIN = True
        self.CLASSES = [1, 2, 3, 4, 5]
        self.INPUT_SIZE = 2
        self.LOG_INTERVAL = 2
        self.DATA_DIM = [10000, 1000]

        self.SPLIT_TRAIN_RATE = 0.8
        self.EPOCHS = 100
        self.BATCH_SIZE_RATE = 0.01

        # Implicit definition of parameters
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.REG:
            self.NUM_CLASSES = 1
        else:
            self.NUM_CLASSES = len(self.CLASSES)
        self.SPLIT_VAL_RATE = (1 - self.SPLIT_TRAIN_RATE) / 2
        self.SPLIT_TEST_RATE = 1 - self.SPLIT_TRAIN_RATE - self.SPLIT_VAL_RATE

        j = 1
        while os.path.isfile(self.SAVE_MODEL_PATH):
            self.SAVE_MODEL_PATH = "./model" + str(j) + ".pth"
            j = j + 1

        torch.manual_seed(self.SEED)

        # Assertions
        assert os.path.isfile(self.DATA_SET_PATH), "DATA_SET_PATH points to no file"
        assert os.path.isfile(self.RES_SET_PATH), "RES_SET_PATH points to no file"

        # Empty fields
        self.BATCH_SIZE = 100
        self.DATASET_SIZE = None

    def set_dataset_size(self, dataset_size):
        self.DATASET_SIZE = dataset_size
        self.BATCH_SIZE = int(self.DATASET_SIZE * self.BATCH_SIZE_RATE)
        return


def get_collabDataBunch(parameters, path, train=True):
    df = pd.read_csv(path)
    df[['userId', 'movieId']] = df['Id'].str.split("_", n=1, expand=True)
    df[['userId']] = df['userId'].str.strip("r")
    df[['movieId']] = df['movieId'].str.strip("c")
    df = df[['userId', 'movieId', 'Prediction']].astype(np.long)
    data = CollabDataBunch.from_df(
        ratings=df,
        valid_pct=parameters.SPLIT_VAL_RATE,
        test=None,
        seed=parameters.SEED,
        path='.',
        bs=parameters.BATCH_SIZE,
        val_bs=None,
        num_workers=defaults.cpus,
        dl_tfms=None,
        device=parameters.DEVICE,
        collate_fn=data_collate,
        no_check=False
    )

    return data


def main():
    parameters = Parameters()
    fastai.device = parameters.DEVICE
    cdb = get_collabDataBunch(parameters, parameters.DATA_SET_PATH)
    y_range = [0.5, 5.5]
    learn = collab_learner(
        cdb,
        n_factors=40,
        y_range=y_range,
        wd=1e-1,
        use_nn=False,
        emb_szs=None,
        layers=None,
        ps=None,
        emb_drop=0.,
        use_bn=True,
        bn_final=False
    )

    learn.fit(parameters.EPOCHS)
    #learn.predict()
    #print(learn.model)

if __name__ == "__main__":
    main()
