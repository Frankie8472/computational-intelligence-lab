import torch
import torch.nn.functional as F

from scipy.signal.filter_design import EPSILON
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from fastai.collab import *
from fastai.metrics import *
import fastai
import torch
import os

if __name__ == '__main__':
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
            self.BATCH_SIZE = 56
            self.DATASET_SIZE = None

        def set_dataset_size(self, dataset_size):
            self.DATASET_SIZE = dataset_size
            self.BATCH_SIZE = int(self.DATASET_SIZE * self.BATCH_SIZE_RATE)
            return


    parameters = Parameters()
    fastai.device = parameters.DEVICE

    df = pd.read_csv(parameters.DATA_SET_PATH)
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
        num_workers=1,
        dl_tfms=None,
        device=parameters.DEVICE,
        collate_fn=data_collate,
        no_check=False
    )

    y_range = [0.5, 5.5]
    learn = collab_learner(
        data,
        n_factors=120,
        y_range=y_range,
        wd=1e-1,
        use_nn=False,
        emb_szs=None,
        layers=None,
        ps=None,
        emb_drop=0.,
        use_bn=True,
        bn_final=False,
        opt_func=AdamW,
        loss_func=MSELossFlat(),  # CrossEntropyFlat, MSELossFlat, BCEFlat, BCEWithLogitsFlat
        metrics=None,
        true_wd=True,
        bn_wd=True,
        train_bn=True,
        path=None,
        model_dir='models',
        silent=False
    )

    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=30, max_lr=1e-2, wd=0.1)

    # learn.export()
    tf = pd.read_csv(parameters.RES_SET_PATH)
    tf[['userId', 'movieId']] = tf['Id'].str.split("_", n=1, expand=True)
    tf[['userId']] = tf['userId'].str.strip("r")
    tf[['movieId']] = tf['movieId'].str.strip("c")
    tf = tf[['userId', 'movieId']].astype(np.long)
    data_test = CollabList.from_df(df=tf, cat_names=['userId', 'movieId'])


    def export_data(data_exp_data, name):
        with open('../output/' + name + 'Submission.csv', 'w') as file:
            file.write('Id,Prediction')
            for idx in range(0, len(data_exp_data)):
                row = int(data_exp_data[idx][0])
                col = int(data_exp_data[idx][1])
                val = data_exp_data[idx][2]
                file.write("\n")
                file.write('r' + str(row) + '_c' + str(col) + ',' + str(val))
        return


    y_pred = np.empty(0)
    for i in range(0, len(tf)):
        if i % 10000 == 0:
            print(str(int(i / 10000)) + " of 117")
        y_pred = np.append(y_pred, learn.predict(tf.iloc[i, :])[1].numpy())

    print(len(tf))
    print(len(y_pred))
    tf['Prediction'] = y_pred

    export_data(tf.to_numpy(), "franz" + str(6))
