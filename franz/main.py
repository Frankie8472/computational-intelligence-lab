from franz.importer import import_data
import numpy as np
from scipy import sparse as sp

csr_matrix = import_data()

# print(csr_matrix.getnnz(0))  # | 10000, wie viele user haben diesen film gesehen, count
print(csr_matrix.getnnz(1))  # -- 1000, wie viele filme hat dieser user geschaut, count
# print(csr_matrix.sum(0).A1)  # | 10000, wie viele user haben diesen film gesehen, bewertungs schnitt
# print(csr_matrix.sum(1).A1)  # -- 1000, wie viele filme hat dieser user geschaut, bewertungs schnitt
# print(np.divide(csr_matrix.sum(1).A1, csr_matrix.getnnz(1)))  # Was für ein Bewertungsdruchschnitt hat der user
print(np.shape(sp.hstack([csr_matrix, np.asmatrix(csr_matrix.getnnz(1)).transpose()]).todense()[:, 1000]))
print(sp.vstack([csr_matrix, csr_matrix.getnnz(0)]).todense().size)
