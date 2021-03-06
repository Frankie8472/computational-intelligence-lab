from scipy.signal.filter_design import EPSILON
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from franz_torch.model import Net, WideAndDeepModel
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from fastai.metrics import accuracy, auc_roc_score


class Parameters:
    def __init__(self):
        # User defined parameters
        self.REG = True
        self.DATA_SET_PATH = "../input/data_train.pt"
        self.RES_SET_PATH = "../output/sampleSubmission.pt"
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
        self.BATCH_SIZE = None
        self.DATASET_SIZE = None

    def set_dataset_size(self, dataset_size):
        self.DATASET_SIZE = dataset_size
        self.BATCH_SIZE = int(self.DATASET_SIZE * self.BATCH_SIZE_RATE)
        self.BATCH_SIZE = 100
        return


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc


def reg_acc(y_pred, y_test):
    correct_pred = torch.abs(y_pred - y_test)
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc


def get_data(params):
    dataset = torch.load(params.DATA_SET_PATH, map_location=params.DEVICE).type(params.DTYPE)
    pred_dataset = torch.load(params.RES_SET_PATH, map_location=params.DEVICE).type(params.DTYPE)

    params.set_dataset_size(len(dataset))

    # Split sets into train, valid and test sets
    train_length = int(len(dataset) * params.SPLIT_TRAIN_RATE)
    valid_length = int(len(dataset) * params.SPLIT_VAL_RATE)
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = random_split(dataset, (train_length, valid_length, test_length))

    # Create data loaders
    train_data_load = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, num_workers=params.WORKERS,
                                 pin_memory=params.PIN)
    valid_data_load = DataLoader(valid_dataset, batch_size=params.BATCH_SIZE, num_workers=params.WORKERS,
                                   pin_memory=params.PIN)
    test_data_load = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, num_workers=params.WORKERS,
                                  pin_memory=params.PIN)

    return train_data_load, valid_data_load, test_data_load


if __name__ == "__main__":
    # Init parameters
    parameters = Parameters()

    # Get data loaders
    train_data_loader, valid_data_loader, test_data_loader = get_data(parameters)

    # Define model
    # model = WideAndDeepModel(parameters.INPUT_SIZE, parameters.NUM_CLASSES).to(device=parameters.DEVICE, dtype=parameters.DTYPE)
    model = WideAndDeepModel(parameters.DATA_DIM, parameters.NUM_CLASSES).to(device=parameters.DEVICE, dtype=parameters.DTYPE)

    # Define loss
    if parameters.REG:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}

    # Train process
    for epoch in range(parameters.EPOCHS):  # 3 full passes over the data
        # Train epoch
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        model.train()

        for i, data in enumerate(train_data_loader, 0):
            X = data[:, 0:-1].long()

            if parameters.REG:
                y = data[:, -1].double()
            else:
                y = data[:, -1].long()
                y = y - 1

            model.zero_grad()
            output = model(X)
            train_loss = criterion(output, y)
            if parameters.REG:
                train_acc = reg_acc(output, y)
            else:
                train_acc = multi_acc(output, y)
            train_loss.backward()
            optimizer.step()

            # print statistics
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            # print(f'Batch {i + 0:03}: | 'f'Train Loss: {loss.item() / len(y):.5f} | 'f'Train Acc: {acc.item() / len(y):.3f}| ')

        # Validate Epoch
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        model.eval()

        correct = 0
        total = 0
        targets, predicts = list(), list()
        with torch.no_grad():
            for data in valid_data_loader:
                X = data[:, 0:-1].long()
                if parameters.REG:
                    y = data[:, -1].double()
                else:
                    y = data[:, -1].long()
                    y = y - 1
                output = model(X)

                val_loss = criterion(output, y)
                if parameters.REG:
                    val_acc = reg_acc(output, y)
                else:
                    val_acc = multi_acc(output, y)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                #_, predicted = torch.max(outputs.data, 1)
                #predicted = predicted + 1

                #targets.extend(y.tolist())
                #predicts.extend(outputs.tolist())

                #total += y.size(0)
                #correct += (predicted == y).sum().item()

        #print('Accuracy of the network: %d %%' % (100 * correct / total))
        #print('ROC of the network: %d %%' % (roc_auc_score(targets, predicts, multi_class="ovo")))

        loss_stats['train'].append(train_epoch_loss / len(train_data_loader))
        loss_stats['val'].append(val_epoch_loss / len(valid_data_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_data_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(valid_data_loader))

        print(
            f'Epoch {epoch + 0:03}: | '
            f'Train Loss: {train_epoch_loss / len(train_data_loader):.5f} | '
            f'Val Loss: {val_epoch_loss / len(valid_data_loader):.5f} | '
            f'Train Acc: {train_epoch_acc / len(train_data_loader):.3f} | '
            f'Val Acc: {val_epoch_acc / len(valid_data_loader):.3f}'
        )

    print('Finished Training')

    '''
    torch.save(model.state_dict(), parameters.SAVE_MODEL_PATH)

    # net = Net()
    # net.load_state_dict(torch.load(PATH))
    # outputs = net(x_pred)
    # _, predicted = torch.max(outputs, 1)

    
    # PREDICT

    y_pred = x_pred
    # FILTER WANTED DATA
    y_pred[x_pred == 0] = 0

    # SAVE DATA
    export_data(y_pred, "franzTorch")
    '''
