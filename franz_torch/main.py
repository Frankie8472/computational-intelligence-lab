import numpy as np
import pandas as pd
import os
import torch
import fastai
from fastai.collab import CollabDataBunch, CollabList, collab_learner, data_collate
from fastai.collab import AdamW, MSELossFlat, load_learner, DatasetType


class Parameters:
    def __init__(self):
        # User defined parameters
        self.EPOCHS = 30
        self.BATCH_SIZE = 56
        self.EMB_SIZE = 128
        self.MAX_LR = 1e-2
        self.WEIGHT_DECAY = 1e-1
        self.SPLIT_VAL_RATE = 0.2

        self.DATA_DIM = [10000, 1000]
        self.REG = True
        self.WORKING_PATH = "."
        self.DATA_SET_PATH = "../input/data_train.csv"
        self.RES_SET_PATH = "../output/sampleSubmission.csv"
        self.MODEL_SAVE_PATH = "./models/"
        self.MODEL_SAVE_NAME = "trained_model.pkl"
        self.DTYPE = torch.double
        self.SEED = 42

        # Implicit definition of parameters
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.WORKERS = 1 if torch.cuda.is_available() else len(os.sched_getaffinity(0))
        torch.manual_seed(self.SEED)

        # Assertions
        assert os.path.isfile(self.DATA_SET_PATH), "DATA_SET_PATH points to no file"
        assert os.path.isfile(self.RES_SET_PATH), "RES_SET_PATH points to no file"


def export_data(data_exp_data):
    i = 0
    filename = '../output/franzSubmission{}.csv'.format(i)

    while os.path.isfile(filename):
        filename = '../output/franzSubmission{}.csv'.format(i)
        i += 1

    with open(filename, 'w') as file:
        file.write('Id,Prediction')
        for idx in range(0, len(data_exp_data)):
            row = int(data_exp_data[idx][0])
            col = int(data_exp_data[idx][1])
            val = data_exp_data[idx][2]
            file.write("\n")
            file.write('r' + str(row) + '_c' + str(col) + ',' + str(val))
    return


def import_data(path, drop_predictions=False):
    import_data_df = pd.read_csv(path)
    import_data_df[['userId', 'movieId']] = import_data_df['Id'].str.split("_", n=1, expand=True)
    import_data_df[['userId']] = import_data_df['userId'].str.strip("r")
    import_data_df[['movieId']] = import_data_df['movieId'].str.strip("c")
    if drop_predictions:
        import_data_df = import_data_df[['userId', 'movieId']].astype(np.long)
    else:
        import_data_df = import_data_df[['userId', 'movieId', 'Prediction']].astype(np.long)

    return import_data_df


def main():
    parameters = Parameters()
    fastai.device = parameters.DEVICE

    print("\n== Loading Data ==")
    df = import_data(parameters.DATA_SET_PATH, drop_predictions=False)
    tf = import_data(parameters.RES_SET_PATH, drop_predictions=True)

    data = CollabDataBunch.from_df(
        ratings=df,
        valid_pct=parameters.SPLIT_VAL_RATE,
        test=None,
        seed=parameters.SEED,
        path='.',
        bs=parameters.BATCH_SIZE,
        val_bs=None,
        num_workers=parameters.WORKERS,
        dl_tfms=None,
        device=parameters.DEVICE,
        collate_fn=data_collate,
        no_check=False
    )

    y_range = (0.5, 5.5)
    learn = collab_learner(
        data,
        n_factors=parameters.EMB_SIZE,
        y_range=y_range,
        wd=parameters.WEIGHT_DECAY,
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

    print("\n== Start Training ==")
    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=parameters.EPOCHS, max_lr=parameters.MAX_LR, wd=parameters.WEIGHT_DECAY)
    learn.export(parameters.MODEL_SAVE_PATH+parameters.MODEL_SAVE_NAME)
    print("\n== Finished Training ==")

    print("\n== Start Predicting ==")
    learn = load_learner(path=parameters.MODEL_SAVE_PATH, file=parameters.MODEL_SAVE_NAME, test=CollabList.from_df(tf))
    y_pred, _ = learn.get_preds(ds_type=DatasetType.Test)
    print("\n== Finished Predicting ==")

    print("\n== Saving Predictions ==")
    tf['Predictions'] = y_pred.numpy()
    export_data(tf.to_numpy())
    return


if __name__ == '__main__':
    main()
