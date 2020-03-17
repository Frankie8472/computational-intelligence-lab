from franz.porter import import_data, export_data
import numpy as np
from sklearn.experimental import enable_iterative_imputer
enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from franz.feature_enhancer import sparse_svd
#from scipy import sparse as sp
#from scipy.sparse.linalg import svds
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

data = import_data().todense()
data[data == 0] = np.nan
data_pred = IterativeImputer(verbose=2).fit_transform(data)

# data_pred = BiScaler().fit_transform(data)
# data_pred = SoftImpute().fit_transform(data_pred)

# data_pred = KNN(k=3).fit_transform(data)

# data_pred = NuclearNormMinimization().fit_transform(data)

export_data(data_pred, 'franz')


