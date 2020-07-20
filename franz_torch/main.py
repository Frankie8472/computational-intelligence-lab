import numpy as np
import pandas as pd
import os
import torch
import fastai
from fastai.collab import CollabDataBunch, CollabList, collab_learner, data_collate, CollabLearner
from fastai.collab import AdamW, MSELossFlat, load_learner, DatasetType, Learner, OptRange, Collection, Tuple

from franz_torch.model import CloudModel


class Parameters:
    def __init__(self):
        # User defined parameters
        self.EPOCHS = 60
        self.BATCH_SIZE = 64
        self.EMB_SIZE = 512
        self.MAX_LR = 1e-2
        self.WEIGHT_DECAY = 1e-1
        self.SPLIT_VAL_RATE = 0.2
        self.PREDICT = False

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


def prophetic_collab_learner(
        data,
        parameters,
        y_range: Tuple[float, float] = None,
        dropout: float = 0.5,
        output_dim: int = 1,
        embed_dim: int = 128,
        dnn_dims: Collection[int] = (2048, 4096, 512, 1024, 128, 256, 32, 64, 8, 16),
        input_depth: int = 1,
        cnn_dims: Collection[int] = (4, 8, 16, 32),
        conv_kernel_size: Collection[int] = (3, 3),
        pool_kernel_size: Collection[int] = (2, 2),
        **learn_kwargs
) -> Learner:
    u, m = data.train_ds.x.classes.values()

    model = CloudModel(
        field_dims=(len(u), len(m)),
        output_dim=output_dim,
        embed_dim=embed_dim,
        y_range=y_range,
        dropout=dropout,
        dnn_dims=dnn_dims,
        input_depth=input_depth,
        cnn_dims=cnn_dims,
        conv_kernel_size=conv_kernel_size,
        pool_kernel_size=pool_kernel_size
    )
    model.to(device=parameters.DEVICE)
    return CollabLearner(data, model, **learn_kwargs)


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

    print("== Loading Data ==")
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
    # learn = collab_learner(
    #     data,
    #     n_factors=parameters.EMB_SIZE,
    #     y_range=y_range,
    #     # wd=parameters.WEIGHT_DECAY,
    #     use_nn=False,
    #     emb_szs={'user': parameters.EMB_SIZE, 'item': parameters.EMB_SIZE},
    #     layers=[256, 1024, 512, 2048, 1024, 128, 256, 64],
    #     ps=None,
    #     emb_drop=0.2,
    #     use_bn=True,
    #     bn_final=False,
    #     # opt_func=AdamW,
    #     loss_func=MSELossFlat(),  # CrossEntropyFlat, MSELossFlat, BCEFlat, BCEWithLogitsFlat
    #     metrics=None,
    #     true_wd=True,
    #     bn_wd=True,
    #     train_bn=True,
    #     path=None,
    #     model_dir='models',
    #     silent=False
    # )

    learn = prophetic_collab_learner(
        data,
        parameters,
        y_range=y_range,
        dropout=0.2,
        output_dim=1,
        embed_dim=parameters.EMB_SIZE,
        dnn_dims=(128, 256, 64, 32, 8),
        input_depth=1,
        cnn_dims=(4, 8, 16, 32),

        wd=parameters.WEIGHT_DECAY,
        opt_func=AdamW,
        loss_func=MSELossFlat(),  # CrossEntropyFlat, MSELossFlat, BCEFlat, BCEWithLogitsFlat
        metrics=None
    )

    print("== Start Training ==")
    learn.unfreeze()
    # learn.lr_find()
    # learn.recorder.plot()

    learn.fit_one_cycle(cyc_len=parameters.EPOCHS, max_lr=parameters.MAX_LR, wd=parameters.WEIGHT_DECAY)
    learn.export(parameters.MODEL_SAVE_PATH + parameters.MODEL_SAVE_NAME)
    print("== Finished Training ==")

    if parameters.PREDICT:
        print("== Start Predicting ==")
        learn = load_learner(path=parameters.MODEL_SAVE_PATH, file=parameters.MODEL_SAVE_NAME, test=CollabList.from_df(tf))
        y_pred, _ = learn.get_preds(ds_type=DatasetType.Test)
        print("== Finished Predicting ==")

        print("== Saving Predictions ==")
        tf['Predictions'] = y_pred.numpy()
        export_data(tf.to_numpy())
    return


if __name__ == '__main__':
    main()
