
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
    pass


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