import glob
import os
import pandas as pd


def export_data(data: pd.DataFrame):
    i = 0
    filename = './csv_bunch/franzBagging{}.csv'.format(i)

    while os.path.isfile(filename):
        filename = './csv_bunch/franzBagging{}.csv'.format(i)
        i += 1

    data.to_csv(filename, sep=",", index=False)
    return


directoryPath = "./csv_bunch/"
ret = pd.DataFrame()
glued_data = pd.DataFrame()
csv_array = glob.glob(directoryPath + '*.csv')
for file_name in csv_array:
    x = pd.read_csv(file_name, low_memory=False)
    if ret.empty:
        ret = x
    glued_data = pd.concat([glued_data, x['Prediction']], axis=1)

ret['Prediction'] = glued_data.mean(axis=1)
# ret['Prediction'] = glued_data.max(axis=1)
# ret['Prediction'] = glued_data.min(axis=1)
# ret['Prediction'] = glued_data.median(axis=1)
ret.loc[ret['Prediction'] <= 1.0, 'Prediction'] = 1.0
ret.loc[ret['Prediction'] >= 5.0, 'Prediction'] = 5.0

export_data(ret)
