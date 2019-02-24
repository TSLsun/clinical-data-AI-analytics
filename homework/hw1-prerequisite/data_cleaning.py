import os
import numpy as np
import pandas as pd

def data_summary(database_dir, summary_path):
    """To summarize database information.

    Please use `pandas.DataFrame` to summarize database
    information. Header includes `PatientID`, `CaseDate`,
    `CaseTime` and `Modality`. Save the dataframe by
    following `PatientID`, `CaseDate`, `CaseTime` and
    `Modality` in ascending order, also, without index.

    Args:
        database_dir (str): Path to database directory.
            Database directory contains each patient directory.
            For example, database_dir = '../data/patients-samples'.
            Under `database_dir` contains `patient-00000`,
            `patient-00001`, etc.
        summary_path (str): Path to save the summary dataframe.
            Dataframe will save in csv format.
    """
    raw_data = []
    for root, dirs, files in os.walk(database_dir):
        if root[-2:] == 'CT' or root[-2:] == 'MR':
            folder_root = root.replace(database_dir, '')
            raw_data.append(folder_root[1:].split('/'))
            
    df = pd.DataFrame(columns=['PatientID', 'CaseDate', 'CaseTime', 'Modality'])
    for i, data in enumerate(raw_data):
        df.loc[i] = [data[0].split('-')[1],
                     "/".join(data[1][4:14].split('.')), 
                     ":".join(data[1][15:].split('.')), 
                     data[2]]
    df = df.sort_values(by=['PatientID', 'CaseDate', 'CaseTime', 'Modality'])
    filename = summary_path + "summary.csv"
    
    # output to summary.csv 
    df.to_csv(filename, index = False)
    
    return df
#     pass


def get_target_cases_df(summary_df):
    """ Get target cases dataframe.

    Target is defined by following:
        1. Must be modality CT.
        2. If patient have several CTs,
           use the latest one.

    Args:
        summary_df (pandas.core.frame.DataFrame):
            Info summary dataframe.

    Returns:
        target_cases_df (pandas.core.frame.DataFrame):
            Target cases dataframe. Index should be successive.
    """
    pass