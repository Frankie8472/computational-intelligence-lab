from franz_torch.porter import export_data
import torch
import torch.nn as nn
import torch.nn.functional as F

# LOAD DATA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_test = torch.load("data_train.pt", map_location=device).double()
x_pred = torch.load("sampleSubmission.pt", map_location=device).double()


y_pred = x_pred
# FILTER WANTED DATA
y_pred[x_pred == 0] = 0

# SAVE DATA
export_data(y_pred, "franzTorch")

