"""Collaborative Filtering through bagging with already existing predictions.

This program takes all csv files in a folder and takes the mean of each row.
The output is then written into a csv file for handing in.
"""

import glob
import os
import pandas as pd


def export_data(export_data_df: pd.DataFrame):
    """Exports data to csv.

    Saves all data from DataFrame to csv. Filename is saved with an index and
    incremented if a file with the same filname already exists.

    Args:
      export_data_df:
        A DataFrame with columns 'Id' and 'Prediction'.
    """

    # Initialize running variable and filename
    i = 0
    filename = '../output/franzBagging{}.csv'.format(i)

    # Assert filename does not exist
    while os.path.isfile(filename):
        filename = '../output/franzBagging{}.csv'.format(i)
        i += 1

        # Assert ratings are between 1.0 and 5.0
        export_data_df.loc[export_data_df['Prediction'] <= 1.0, 'Prediction'] = 1.0
        export_data_df.loc[export_data_df['Prediction'] >= 5.0, 'Prediction'] = 5.0

        # Export to csv
        export_data_df.to_csv(filename, index=False)
        return


def main():
    """Main function.

    Handles the following task in the following order.
    1. Initializing parameters and empty DataFrames.
    2. Load and save all csv files in the same DataFrame.
    3. Take the mean over each row.
    4. Assert the final values are between 1.0 and 5.0.
    5. Export new calculated predictions.
    """

    # Initializing parameters and empty DataFrames
    directoryPath = "./csv_bunch/"
    csv_array = glob.glob(directoryPath + '*.csv')
    ret = pd.DataFrame()
    glued_data = pd.DataFrame()

    # Load and save all csv files in the same DataFrame
    for file_name in csv_array:
        x = pd.read_csv(file_name, low_memory=False)
        if ret.empty:
            ret = x
        glued_data = pd.concat([glued_data, x['Prediction']], axis=1)

    # Take the mean over each row
    ret['Prediction'] = glued_data.mean(axis=1)  # max(axis=1), min(axis=1), median(axis=1)

    # Assert the final values are between 1.0 and 5.0
    ret.loc[ret['Prediction'] <= 1.0, 'Prediction'] = 1.0
    ret.loc[ret['Prediction'] >= 5.0, 'Prediction'] = 5.0

    # Export new calculated predictions
    export_data(ret)
    return


# Make sure the executed file is this one
if __name__ == "__main__":
    main()
