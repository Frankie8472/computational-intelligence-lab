"""Collaborative Filtering through the embedding-dotproduct.

With the help of the fastai library based on pytorch, a
learner is trained to predict ratings for user-movie indexes.
Necessary parameters are initialized at the beginning with a
parameter class. The train and test data are imported/exported
with the help of the pandas library and stored in a dataframe.
The learner is trained through the one cycle policy in batches.
If an optimum is found, the user can

  Typical usage example:

  1. Set parameters in class Parameters
  2. Run
"""

import numpy as np
import pandas as pd
import os
import torch
import fastai
from fastai.collab import CollabDataBunch, CollabList, collab_learner
from fastai.collab import AdamW, MSELossFlat, load_learner, DatasetType


class Parameters:
    """Class containing all parameters used in this file.

    All parameters should only be changed here, so one does not have to search in the code below.

    Attributes:
        CYCLES: An integer as number of repeating "one-cycle-policy-training".
        EPOCHS: An integer as number of epochs to train the whole set with "one-cycle-policy".
        BATCH_SIZE: An integer as size of the train set.
        EMB_SIZE: An integer as size of the embedding vectors.
        MAX_LR: A double as highest learning rate used in the "one-cycle-policy".
        WEIGHT_DECAY: A double as weight decay used in "one-cycle-policy".
        SPLIT_VAL_RATE: A double as size of the validation set in correlation to the whole training set.
        PREDICT: A boolean if a prediction should be made of the test set. SPLIT_VAL_RATE will automaticly be set to zero.
        USE_DNN: A boolean to use a DNN instead of the DOT Product (do not change this).
        WORKING_PATH: A string with the working path.
        DATA_SET_PATH: A string with the full path to the train set.
        RES_SET_PATH: A string with the full path to the test set.
        MODEL_SAVE_PATH: A string with the path to the model save folder.
        MODEL_SAVE_NAME: A string with the name of the saved model (model to be saved).
        DTYPE: DTYPE of the torch library (do not change this).
        SEED: A integer as seed.
        DEVICE: A string containing the name of the device this code runs on (gpu or cpu, automatic, do not change this).
        WORKERS: An integer as the number of cores used in the execution (automatic, do not change this).
    """

    def __init__(self):
        """Inits ParameterClass with predefined parameters."""

        # User defined parameters for the learner
        self.CYCLES = 1
        self.EPOCHS = 18
        self.BATCH_SIZE = 100
        self.EMB_SIZE = 128
        self.MAX_LR = 1e-2
        self.WEIGHT_DECAY = 1e-1
        self.PREDICT = False

        # Parameters which should not have to be changed
        self.SPLIT_VAL_RATE = 0.1  # Automatically set to zero if self.Predict set to True
        self.USE_DNN = False
        self.WORKING_PATH = "."
        self.DATA_SET_PATH = "../input/data_train.csv"
        self.RES_SET_PATH = "../output/sampleSubmission.csv"
        self.MODEL_SAVE_PATH = "./models/"
        self.MODEL_SAVE_NAME = "trained_model.pkl"
        self.DTYPE = torch.double
        self.SEED = 42

        # Automatic detection of the most suitable device
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Enable multithreading if run on the cpu
        self.WORKERS = 1 if torch.cuda.is_available() else len(os.sched_getaffinity(0))
        torch.manual_seed(self.SEED)

        # If the classifier is set on predicting, the classifier should be trained on the whole set
        if self.PREDICT:
            self.SPLIT_VAL_RATE = 0.0

        # File existence assertions
        assert os.path.isfile(self.DATA_SET_PATH), "DATA_SET_PATH points to no file"
        assert os.path.isfile(self.RES_SET_PATH), "RES_SET_PATH points to no file"


def import_data(path: str, drop_predictions: bool = False):
    """Imports data from csv.

    Retrieves all data from csv, puts it into pandas dataframe and converts
    string id to integer row and column index.

    Args:
      path:
        A string with the full path and filename to the file to be imported.
      drop_predictions:
        Optional; If drop_predictions is True, rating column is dropped.

    Returns:
      Two pandas.DataFrame. The first one for training/predicting, the second
      one for exporting predicted values by replacing the ratings collumn with
      the predicted values.

      If drop_predictions is False, second DataFrame has no use here

      For train example:
      original_tf['Prediction'] = y_pred.numpy()
    """

    # Import csv
    original_id_df = pd.read_csv(path, low_memory=False)

    # Preserve original column entries for export purposes
    import_data_df = original_id_df.copy()

    # Transform string encrypted indexes to usable integer values
    import_data_df[['userId', 'movieId']] = import_data_df['Id'].str.split("_", n=1, expand=True)
    import_data_df[['userId']] = import_data_df['userId'].str.strip("r")
    import_data_df[['movieId']] = import_data_df['movieId'].str.strip("c")

    # Drop ratings column if it is the test file
    if drop_predictions:
        import_data_df = import_data_df[['userId', 'movieId']].astype(np.long)
    else:
        import_data_df = import_data_df[['userId', 'movieId', 'Prediction']].astype(np.long)

    return import_data_df, original_id_df


def export_data(export_data_df: pd.DataFrame, e, bs, emb):
    """Exports data to csv.

    Saves all data from DataFrame to csv. Filename is saved with an index and
    incremented if a file with the same filname already exists.

    Args:
      export_data_df:
        A DataFrame with columns 'Id' and 'Prediction'.
    """

    # Initialize running variable and filename
    i = 0
    filename = '../output/franzSubmission:bs-{}_e-{}_emb-{}:"{}.csv'.format(e, bs, emb, i)

    # Assert filename does not exist
    while os.path.isfile(filename):
        filename = '../output/franzSubmission:bs-{}_e-{}_emb-{}:"{}.csv'.format(e, bs, emb, i)
        i += 1

    # Assert ratings are between 1.0 and 5.0
    export_data_df.loc[export_data_df['Prediction'] <= 1.0, 'Prediction'] = 1.0
    export_data_df.loc[export_data_df['Prediction'] >= 5.0, 'Prediction'] = 5.0

    # Export to csv
    export_data_df.to_csv(filename, index=False)
    return


def main(e, bs, emb, cyc):
    """Main function.

    Handles the following task in the following order.
    1. Initializes parameters and chooses ideal device.
    2. Imports train and test data.
    3. Splits train data into train and valid set in batches.*
    4. Initializes an learner based on embedding and dotproduct.
    5. Trains the learner with the one cycle policy.
    6. Exports the model.
    7. Imports the model for predicting purposes.**
    8. Predicting values for test set**
    9. Exports predicted values to csv.**

    *   Valid set is empty if predicting is enabled in the parameter class.
    **  Only executed if predicting is enabled in the parameter class.
    """

    # Initializing parameters and ideal device
    parameters = Parameters()
    fastai.device = parameters.DEVICE

    print("== Device Selection ==")
    print("Device: {}".format(parameters.DEVICE))

    # Import train and test data
    print("== Loading Data ==")
    df, _ = import_data(parameters.DATA_SET_PATH, drop_predictions=False)
    tf, original_tf = import_data(parameters.RES_SET_PATH, drop_predictions=True)

    # Split train set into train and validation set in batches
    data = CollabDataBunch.from_df(
        ratings=df,
        valid_pct=parameters.SPLIT_VAL_RATE,
        seed=parameters.SEED,
        path='.',
        bs=bs,
        num_workers=parameters.WORKERS,
        device=parameters.DEVICE,
    )

    # Optimiztaion trick by extending rating boundaries
    y_range = (0.5, 5.5)

    # Initializing learner
    learn = collab_learner(
        data,
        n_factors=emb,
        y_range=y_range,
        wd=parameters.WEIGHT_DECAY,
        use_nn=parameters.USE_DNN,
        opt_func=AdamW,
        loss_func=MSELossFlat()
    )

    # Start training cycle
    print("== Start Training ==")

    # Make all parameters trainable
    learn.unfreeze()

    # Loop over cycles
    for cycle in range(0, cyc):
        # Train model with one cylce policy (increasing and decreasing learning rate)
        learn.fit_one_cycle(cyc_len=e, max_lr=parameters.MAX_LR, wd=parameters.WEIGHT_DECAY)

    # Export trained model
    learn.export(parameters.MODEL_SAVE_PATH + parameters.MODEL_SAVE_NAME)

    print("== Finished Training ==")

    # Start prediction
    if parameters.PREDICT:
        print("== Start Predicting ==")

        # Load trained model
        learn = load_learner(path=parameters.MODEL_SAVE_PATH, file=parameters.MODEL_SAVE_NAME,
                             test=CollabList.from_df(tf))

        # Predict test set
        y_pred, _ = learn.get_preds(ds_type=DatasetType.Test)

        print("== Finished Predicting ==")
        print("== Saving Predictions ==")

        # Export predicted values for test set in csv
        original_tf['Prediction'] = y_pred.numpy()
        export_data(original_tf, e, bs, emb)
    return


# Make sure the executed file is this one
if __name__ == '__main__':
    for e in [3, 4, 5]:
        for bs in [20000, 30000, 50000, 70000]:
            for emb in [8, 30, 100, 120, 160, 200, 220]:
                print(":::bs-{}_e-{}_emb-{}:::".format(e, bs, emb))
                main(bs, e, emb, cyc=2)
