# from cuml.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from monai.transforms import EnsureChannelFirstd
import numpy as np
import monai
import torchvision


TRANSFORMATION_MODES = {'CT': 'bilinear', 'CT_target': 'bilinear', 'RTDOSE': 'bilinear', 'RTDOSE_target': 'bilinear',
                        'RTSTRUCT': 'nearest', 'RTSTRUCT_target': 'nearest'}

class StandardScalerd(object):
    # another possibility is to compute the mean and standard deviation only using partial_fit
    def __init__(self, keys, copy=True, with_mean=True, with_std=True, continuous_variables=None):
        self.keys = keys
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.continuous_variables = continuous_variables
        self.continuous_vars_indexes = None
        self.transformer = {k: StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std) for k in keys}

    def __call__(self, data_dict):
        # here comes in a dictionary with numpy arrays or tensors
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k][self.continuous_vars_indexes] = self.transformer[k].transform(
                    data_dict[k][[self.continuous_vars_indexes]])
        return data_dict

    def fit(self, data_pd):
        # here comes in a pandas dataframe with tabular data
        if self.continuous_variables is None:
            self.continuous_variables = data_pd.columns
        self.continuous_vars_indexes = data_pd.columns.get_indexer(self.continuous_variables)
        for k in self.keys:
            assert len(list(data_pd.shape)) == 2
            self.transformer[k].fit(data_pd.loc[:, self.continuous_variables].values)


def transform_pipeline(config, rd=None):
    img_keys = [k for k in config['MODALITY'].keys() if config['MODALITY'][k]]
    records_keys = ['records'] if config['RECORDS']['records'] else []

    if len(records_keys) > 0 or len(img_keys) > 0:
        train_transform = []
        val_transform = []

        if len(records_keys) > 0:
            if 'continuous_cols' not in config['DATA'].keys():
                non_continuous = [config['DATA']['target'], config['DATA']['censor_label'],
                                  config['DATA']['subject_label']]
                config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols']
                                                     if col not in non_continuous]
            train_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]
            val_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]

        if len(img_keys) > 0:
            reduced_keys = list(set(img_keys).difference(set(['RTDOSE'])))
            reduced_keys = reduced_keys if len(reduced_keys) > 0 else ['dummy']
            condition = (('RTSTRUCT' not in config['MODALITY'].keys()) or (not config['MODALITY']['RTSTRUCT']) and
                         (config['MODALITY']['CT']) and ('CT' in config['MODALITY'].keys()))
            train_transform = [
                EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys, allow_missing_keys=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=-1024, a_max=1500, b_min=0, b_max=1, keys=['CT'], allow_missing_keys=True, clip=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=0, a_max=80, b_min=0, b_max=1, keys=['RTDOSE'], allow_missing_keys=True, clip=True),
                monai.transforms.RandHistogramShiftd(keys=['CT'], allow_missing_keys=True, prob=0.2),
                monai.transforms.RandAdjustContrastd(keys=['CT'], allow_missing_keys=True, gamma=(0.7, 3), prob=0.2),
                monai.transforms.RandGaussianNoised(keys=['CT'], allow_missing_keys=True, std=0.07, prob=0.2),
                monai.transforms.RandAffined(keys=img_keys, allow_missing_keys=True, rotate_range=(0., 0., 0.3),
                                             scale_range=(0.2, 0.2, 0.2), padding_mode='zeros', prob=0.3,
                                             mode=[TRANSFORMATION_MODES[elem] for elem in img_keys]),
            ]

            val_transform = [
                EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys, allow_missing_keys=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=-1024, a_max=1500, b_min=0, b_max=1, keys=['CT'], allow_missing_keys=True, clip=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=0, a_max=80, b_min=0, b_max=1, keys=['RTDOSE'], allow_missing_keys=True, clip=True),
            ]

        if rd is not None:
            for j, tr in enumerate(train_transform):
                if hasattr(tr, 'set_random_state'):
                    train_transform[j].set_random_state(seed=rd)

        train_transform = torchvision.transforms.Compose(train_transform)
        val_transform = torchvision.transforms.Compose(val_transform)
    else:
        train_transform = None
        val_transform = None

    return train_transform, val_transform


def reconstruction_transform_pipeline(config, rd=None):
    img_keys = [k for k in config['MODALITY'].keys() if config['MODALITY'][k]]
    records_keys = ['records'] if config['RECORDS']['records'] else []
    label_keys = [f'{elem}_target' for elem in config['DATA']['reconstruction_target']]

    if len(records_keys) > 0 or len(img_keys) > 0:
        train_transform = []
        val_transform = []

        if len(records_keys) > 0:
            if 'continuous_cols' not in config['DATA'].keys():
                non_continuous = [config['DATA']['target'], config['DATA']['censor_label'],
                                  config['DATA']['subject_label']]
                config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols']
                                                     if col not in non_continuous]
            train_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]
            val_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]

        if len(img_keys) > 0:
            train_transform = [
                EnsureChannelFirstd(keys=img_keys+label_keys, allow_missing_keys=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=-1024, a_max=1500, b_min=0, b_max=1, keys=['CT', 'CT_target'], allow_missing_keys=True, clip=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=0, a_max=80, b_min=0, b_max=1, keys=['RTDOSE', 'RTDOSE_target'], allow_missing_keys=True, clip=True),
                monai.transforms.RandGaussianNoised(keys=['CT'], allow_missing_keys=True, std=0.07, prob=0.2),
                monai.transforms.RandAffined(keys=img_keys+label_keys, allow_missing_keys=True,
                                             rotate_range=(0., 0., 0.3), scale_range=(0.2, 0.2, 0.2),
                                             padding_mode='zeros', prob=0.99,
                                             mode=[TRANSFORMATION_MODES[elem] for elem in img_keys+label_keys]),
            ]

            val_transform = [
                EnsureChannelFirstd(keys=img_keys+label_keys, allow_missing_keys=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=-1024, a_max=1500, b_min=0, b_max=1, keys=['CT', 'CT_target'], allow_missing_keys=True, clip=True),
                monai.transforms.ScaleIntensityRanged(
                    a_min=0, a_max=80, b_min=0, b_max=1, keys=['RTDOSE', 'RTDOSE_target'], allow_missing_keys=True, clip=True),
                # monai.transforms.RandGaussianNoised(keys=['CT'], allow_missing_keys=True, std=0.07, prob=0.2),
                # monai.transforms.RandAffined(keys=img_keys, allow_missing_keys=True,
                #                              rotate_range=(0., 0., 0.3), scale_range=(0.2, 0.2, 0.2),
                #                              padding_mode='zeros', prob=0.99,
                #                              mode=[TRANSFORMATION_MODES[elem] for elem in img_keys]),
            ]

        if rd is not None:
            for j, tr in enumerate(train_transform):
                if hasattr(tr, 'set_random_state'):
                    train_transform[j].set_random_state(seed=rd)

        train_transform = torchvision.transforms.Compose(train_transform)
        val_transform = torchvision.transforms.Compose(val_transform)
    else:
        train_transform = None
        val_transform = None

    return train_transform, val_transform
