import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def create_subject_list(config):
    patients = pd.read_csv(config['DATA']['patient_ids_file_path'])['PatientID']
    data_columns = config['DATA']['clinical_cols'] + [config['DATA']['target']] + config['DATA']['additional_targets']
    if 'censor_label' in config['DATA'].keys():
        data_columns.append(config['DATA']['censor_label'])
    subject_list = pd.read_csv(config['DATA']['clinical_table_path'])
    subject_list.loc[:, 'PatientPathSuffix'] = subject_list.loc[:, 'PatientID']
    subject_list.loc[:, 'PatientID'] = subject_list.loc[:, 'PatientID'].apply(lambda x: Path(x).parts[-1])
    subject_list = subject_list.set_index('PatientID')
    patient_paths = [Path(config['DATA']['data_folder']) / subject_list.loc[pat, 'PatientPathSuffix']
                     for pat in patients if pat in subject_list.index]
    subject_list = subject_list.loc[subject_list.index.isin(patients), data_columns]
    subject_list = subject_list.loc[[p.stem for p in patient_paths]]
    # Certify censored value is 0 and event 1
    if 'censor_label' in config['DATA'].keys() and 'censored_value' in config['DATA'].keys():
        subject_list['Censored'] = (
                subject_list[config['DATA']['censor_label']] == config['DATA']['censored_value']).astype(float)
    elif 'censor_label' not in config['DATA'].keys():
        subject_list['Censored'] = 0

    if config['DATA']['censored_as_nan']:
        subject_list.loc[subject_list['Censored'].astype(bool), config['DATA']['target']] = np.nan

    # Add each patient's modality paths
    for modality in config['MODALITY']:
        if config['MODALITY'][modality] or (
                'reconstruction_target' in config['DATA'] and modality in config['DATA']['reconstruction_target']):
            filename = config['DATA'][f'{modality}_path']
            subject_list[f'{modality}_Path'] = [pat / filename for pat in patient_paths]

    if config['DATA']['impute_addit_target']:
        for i, col in enumerate(config['DATA']['additional_targets']):
            imputer = (SimpleImputer() if config['MODEL']['modes'][i + 1] == 'regression' else
                       SimpleImputer(strategy='most_frequent'))
            subject_list.loc[:, [col]] = imputer.fit_transform(subject_list.loc[:, [col]]).astype(np.float32)

    if config['DATA']['exclude_event_nan']:
        subject_list = subject_list.loc[
            subject_list[[config['DATA']['target']] +
                         config['DATA']['additional_targets']].sum(axis=1, skipna=False).notna()]

    if not config['DATA']['include_censored']:
        subject_list = subject_list.loc[subject_list.loc[:, 'Censored'] == 0]

    if 'threshold' in config['DATA']:  # it's classification
        # Exclude all patients with censored times before the threshold
        exclusion_cond = ((subject_list.loc[:, 'Censored'] == 1) &
                          (subject_list.loc[:, config['DATA']['target']] < config['DATA']['threshold']))
        subject_list = subject_list.loc[~exclusion_cond]

    if 'RECORDS' in config.keys() and config['RECORDS']['records'] and 'categorical_cols' in config['DATA']:
        subject_list_old = subject_list.copy()
        if config['DATA']['one_hot_enc']:
            subject_list = pd.get_dummies(subject_list, columns=config['DATA']['categorical_cols'], drop_first=True,
                                          dtype=np.float32)
            new_categorical_cols = []
            for var in config['DATA']['categorical_cols']:
                cols = [c for c in list(subject_list.columns) if var in c]
                subject_list.loc[subject_list_old[var].isna(), cols] = np.nan
                new_categorical_cols += cols
            config['DATA']['given_categorical_cols'] = config['DATA']['categorical_cols']
            config['DATA']['given_clinical_cols'] = config['DATA']['clinical_cols']
            config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols'] if col not in config['DATA']['categorical_cols']]
            config['DATA']['categorical_cols'] = new_categorical_cols
            config['DATA']['clinical_cols'] = config['DATA']['continuous_cols'] + config['DATA']['categorical_cols']
        else:
            config['DATA']['given_categorical_cols'] = config['DATA']['categorical_cols']
            config['DATA']['given_clinical_cols'] = config['DATA']['clinical_cols']
            config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols'] if
                                                 col not in config['DATA']['categorical_cols']]
            config['DATA']['clinical_cols'] = config['DATA']['continuous_cols'] + config['DATA']['categorical_cols']

    if config['DATA']['imputation'] == 'mice' and len(config['DATA']['clinical_cols']) > 0:
        imputer = IterativeImputer()
        subject_list.loc[:, config['DATA']['clinical_cols']] = imputer.fit_transform(
            subject_list.loc[:, config['DATA']['clinical_cols']]).astype(np.float32)
    elif config['DATA']['imputation'] == 'none':
        cardinalities = eval(config['DATA']['cat_cardinalities'])
        for col in config['DATA']['categorical_cols']:
            subject_list.loc[:, col] = add_missing_as_category(subject_list[col], cardinalities[col])

    return subject_list.reset_index()


def add_missing_as_category(
    values: pd.Series,
    cardinality: Optional[int] = None,
) -> pd.Series:
    """
    Replace missing values in an int-coded categorical variable with a new category,
    using 0 as the code for missing and shifting all existing category codes up by 1.

    Parameters
    ----------
    values : pandas Series
        Input data: pandas Series.
        Values should be integer codes (possibly with missing values).

    cardinality : int, optional
        Expected total number of categories NOT INCLUDING the missing category.
        If provided, this function will check that all resulting codes are
        strictly less than `cardinality` and raise an error otherwise.

        This is typically the value you would pass to `num_embeddings` in a
        PyTorch `nn.Embedding` (i.e. max index + 1).

    Returns
    -------
    pandas Series
        The input with:
        - missing values coded as 0
        - all original non-missing codes increased by 1
    """
    ser = values.copy()
    missing_mask = ser.isna()
    non_missing_mask = ~missing_mask

    # Optional consistency check if cardinality is provided
    if cardinality is not None:
        max_code = int(ser.max())
        if max_code >= cardinality:
            raise ValueError(
                f"Resulting codes go up to {max_code}, which is incompatible "
                f"with cardinality={cardinality}."
            )

    # Shift all existing (non-missing) category codes up by 1
    if non_missing_mask.any():
        ser.loc[non_missing_mask] = ser.loc[non_missing_mask].astype("int64") + 1

    # Assign 0 to missing values
    ser.loc[missing_mask] = 0

    # No missing values left, so we can safely use an integer dtype
    ser = ser.astype("int64")

    return ser
