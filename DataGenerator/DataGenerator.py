from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from collections import Counter
from monai.visualize.utils import matshow3d
from monai.transforms import LoadImage, EnsureChannelFirstd, ResampleToMatchd, ResizeWithPadOrCropd
from monai.data.meta_tensor import MetaTensor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from copy import deepcopy
import time
import SimpleITK as sitk
import nibabel as nib
import random


def timeit_decorator(repeats=1000):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            for _ in range(repeats):
                result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed_time = (end - start) / repeats
            print(f"Average execution `get_item` time over {repeats} runs: {elapsed_time:.6f} seconds")
            return result

        return wrapper

    return decorator


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, SubjectList, config=None, keys=['CT'], transform=None, inference=False, clinical_cols=None,
                 session=None, predict=False, **kwargs):
        super().__init__()
        self.config = config
        self.SubjectList = SubjectList
        self.keys = [k for k in keys if config['MODALITY'][k]]
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols
        self.predict = predict
        self.n = 0
        self.times = []

    def __len__(self):
        return int(self.SubjectList.shape[0])

    def __getitem__(self, i):
        data = {}
        meta = {}
        data['slabel'] = self.SubjectList.loc[i, self.config['DATA']['subject_label']]
        ## Load CT
        if 'CT' in self.keys and self.config['MODALITY']['CT']:
            CTPath = self.SubjectList.loc[i, 'CT_Path']
            CT_Path = Path(CTPath, 'CT.nii.gz')
            data['CT'] = LoadImage(image_only=True, reader='ITKReader')(CT_Path)

        ## Load RTDOSE
        if 'RTDOSE' in self.keys and self.config['MODALITY']['RTDOSE']:
            RTDOSEPath = self.SubjectList.loc[i, 'RTDOSE_Path']
            RTDOSEPath = Path(RTDOSEPath, 'Dose.nii.gz')
            data['RTDOSE'] = LoadImage(image_only=True)(RTDOSEPath)

        ## Load PET
        if 'PET' in self.keys and self.config['MODALITY']['PET']:
            PETPath = self.SubjectList.loc[i, 'PET_Path']
            if self.config['DATA']['Nifty']:
                PETPath = Path(PETPath, 'pet.nii.gz')
            data['PET'] = LoadImage(image_only=True)(PETPath)

        ## Load Mask
        if 'RTSTRUCT' in self.keys and self.config['MODALITY']['RTSTRUCT']:
            RSPath = self.SubjectList.loc[i, 'RTSTRUCT_Path']
            RS_Path = Path(RSPath, self.config['DATA']['structs'] + '.nii.gz')
            data['RTSTRUCT'] = LoadImage(image_only=True)(RS_Path)
            data['RTSTRUCT'][data['RTSTRUCT'] > 0] += 3

        # Add clinical record at the end
        if 'RECORDS' in self.config.keys() and self.config['RECORDS']['records']:
            data['records'] = self.SubjectList.loc[i, self.clinical_cols].values.astype('float')

        if self.transform:
            data = self.transform(data)

        if self.config['DATA']['multichannel']:
            old_keys = list(self.keys)
            data['Image'] = np.concatenate([data[key] for key in old_keys], axis=0)
            for key in old_keys: data.pop(key)
        else:
            if 'RTSTRUCT' in data.keys() and 'RTDOSE' in data.keys():
                data['RTDOSE'] = np.concatenate([data[key] for key in ['RTDOSE', 'RTSTRUCT']], axis=0)
                data.pop('RTSTRUCT')
            elif 'RTSTRUCT' in data.keys():
                data.pop('RTSTRUCT')  # No need for mask in single-channel multi-branch

        if self.inference:
            return data
        else:
            label = np.array(
                self.SubjectList.loc[i, [self.config['DATA']['target']] + self.config['DATA']['additional_targets']],
                dtype=np.float32)
            if self.config['MODEL']['modes'][0] == 'classification':  # Classification
                label[0] = np.where(label[0] > self.config['DATA']['threshold'], 1, 0)
                label[0] = torch.as_tensor(label[0], dtype=torch.float32)
            if 'censor_label' in self.config['DATA'].keys():
                censor_status = np.float32(self.SubjectList.loc[i, 'Censored']).astype('bool')
                if self.config['DATA']['censored_value'] == 1:
                    censor_status = ~censor_status
                return (data, label, censor_status, i) if self.predict else (data, label, censor_status, i)
            else:
                return (data, label, i) if self.predict else (data, label, i)


class DataGeneratorReconstruction(torch.utils.data.Dataset):
    def __init__(self, SubjectList, config=None, keys=('CT',), transform=None, inference=False, clinical_cols=None,
                 predict=False):
        super().__init__()
        self.config = config
        self.SubjectList = SubjectList
        self.keys = [k for k in keys if config['MODALITY'][k]]
        self.label_keys = self.config['DATA']['reconstruction_target']
        self.transform = transform
        self.inference = inference
        self.clinical_cols = clinical_cols
        self.predict = predict
        self.n = 0
        self.times = []

    def __len__(self):
        return int(self.SubjectList.shape[0])

    def __getitem__(self, i):
        data = {}
        data['slabel'] = self.SubjectList.loc[i, self.config['DATA']['subject_label']]
        ## Load CT
        if 'CT' in self.keys or 'CT' in self.label_keys:
            CTPath = self.SubjectList.loc[i, 'CT_Path']
            CT_Path = Path(CTPath, 'CT.nii.gz')
            ct = LoadImage(image_only=True, reader='ITKReader')(CT_Path)
            if 'CT' in self.keys:
                data['CT'] = ct
            if 'CT' in self.label_keys:
                data['CT_target'] = ct

        ## Load RTDOSE
        if 'RTDOSE' in self.keys or 'RTDOSE' in self.label_keys:
            RTDOSEPath = self.SubjectList.loc[i, 'RTDOSE_Path']
            RTDOSEPath = Path(RTDOSEPath, 'Dose.nii.gz')
            dose = LoadImage(image_only=True)(RTDOSEPath)
            if 'RTDOSE' in self.keys:
                data['RTDOSE'] = dose
            if 'RTDOSE' in self.label_keys:
                data['RTDOSE_target'] = dose

        ## Load PET
        if 'PET' in self.keys or 'PET' in self.label_keys:
            PETPath = self.SubjectList.loc[i, 'PET_Path']
            if self.config['DATA']['Nifty']:
                PETPath = Path(PETPath, 'pet.nii.gz')
            pet = LoadImage(image_only=True)(PETPath)
            if 'PET' in self.keys:
                data['PET'] = pet
            if 'PET' in self.label_keys:
                data['PET_target'] = pet

        ## Load Mask
        if 'RTSTRUCT' in self.keys or 'RTSTRUCT' in self.label_keys:
            RSPath = self.SubjectList.loc[i, 'RTSTRUCT_Path']
            RS_Path = Path(RSPath, self.config['DATA']['structs'] + '.nii.gz')
            struct = LoadImage(image_only=True)(RS_Path)
            struct[struct>0] = 1
            if 'RTSTRUCT' in self.keys:
                data['RTSTRUCT'] = struct
            if 'RTSTRUCT' in self.label_keys:
                data['RTSTRUCT_target'] = struct

        # Add clinical record at the end
        if ('RECORDS' in self.config.keys() and self.config['RECORDS']['records']) or 'RECORDS' in self.label_keys:
            records = self.SubjectList.loc[i, self.clinical_cols].values.astype('float')
            if 'RECORDS' in self.keys:
                data['records'] = records
            if 'RECORDS' in self.label_keys:
                data['records_target'] = records

        if self.transform:
            data = self.transform(data)

        data['Image'] = np.concatenate([data[key] for key in self.keys], axis=0)
        label = np.concatenate([data[f'{key}_target'] for key in self.label_keys], axis=0)
        for key in self.keys+[f'{k}_target' for k in self.label_keys]:
            data.pop(key)

        if self.inference:
            return data
        else:
            return data, label


# DataLoader
class DataModule(LightningDataModule):
    def __init__(self, SubjectList, config=None, train_transform=None, val_transform=None, train_size=0.7,
                 train_fraction=1., rd=None, rd_worker=None, num_workers=1, prefetch_factor=None, split_list=None,
                 reconstruction=False, **kwargs):
        super().__init__()
        self.batch_size = config['MODEL']['batch_size']
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.rd = rd
        self.rd_worker = rd_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(int(self.rd_worker))
        self.train_fraction = train_fraction

        train_list, val_list, test_list = self.get_train_val_test(
            config, rd, train_size, SubjectList, train_fraction, split_list)

        train_transform = self.transform_fit(train_transform, train_list, config)
        val_transform = self.transform_fit(val_transform, train_list, config)

        self.train_list = train_list.reset_index(drop=True)
        self.val_list = val_list.reset_index(drop=True)
        self.test_list = test_list.reset_index(drop=True)
        self.full_list = SubjectList.reset_index(drop=True)

        # scale targets
        if not reconstruction:
            self.target_preprocessing = dict()
            for i, col in enumerate([config['DATA']['target']] + config['DATA']['additional_targets']):
                t_preproc = RobustScaler() if config['MODEL']['modes'][i] == 'regression' else MinMaxScaler()
                self.train_list.loc[:, [col]] = t_preproc.fit_transform(self.train_list.loc[:, [col]]).astype(np.float32)
                self.val_list.loc[:, [col]] = t_preproc.transform(self.val_list.loc[:, [col]]).astype(np.float32)
                self.test_list.loc[:, [col]] = t_preproc.transform(self.test_list.loc[:, [col]]).astype(np.float32)
                self.full_list.loc[:, [col]] = t_preproc.transform(self.full_list.loc[:, [col]]).astype(np.float32)
                self.target_preprocessing[col] = t_preproc

        data_generator = DataGeneratorReconstruction if reconstruction else DataGenerator
        self.train_data = data_generator(self.train_list, config=config, transform=train_transform, **kwargs)
        self.val_data = data_generator(self.val_list, config=config, transform=val_transform, **kwargs)
        self.test_data = data_generator(self.test_list, config=config, transform=val_transform, **kwargs)
        self.full_data = data_generator(self.full_list, config=config, transform=val_transform, predict=True, **kwargs)

        self.train_sampler = RandomSampler(self.train_data, generator=self.generator)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.full_data, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def get_train_val_test(self, config, rd, train_size, subject_list, train_fraction=1., split_list=None):
        if split_list is not None:
            train_list = subject_list.loc[subject_list['PatientID'].isin(
                split_list.loc[split_list['Train Set'], 'PatientID'])]
            val_list = subject_list.loc[subject_list['PatientID'].isin(
                split_list.loc[split_list['Validation Set'], 'PatientID'])]
            test_list = subject_list.loc[subject_list['PatientID'].isin(
                split_list.loc[split_list['Test Set'], 'PatientID'])]
            return train_list, val_list, test_list
        if config['RUN']['cross_validation']:
            train_list, val_list, test_list = self._cv_(
                subject_list, train_size, config['RUN']['cv_k'], config['RUN']['cv_fold'], rd)
        else:
            train_list, val_list, test_list = self._random_split_(
                subject_list, train_size, rd)
        if train_fraction != 1.:
            idxs_to_keep = random.sample(list(train_list.index.values), int(train_list.shape[0] * train_fraction))
            train_list = train_list.loc[idxs_to_keep]
        return train_list, val_list, test_list

    def _cv_(self, subject_list, train_size, k, fold, rd):
        train_list, val_list, test_list = self._random_split_(subject_list, train_size, rd)
        full_train = pd.concat([train_list, val_list], axis=0)
        k_splitter = KFold(k, shuffle=True, random_state=rd)
        k_folds = list(k_splitter.split(full_train))
        return full_train.iloc[k_folds[fold][0]], full_train.iloc[k_folds[fold][1]], test_list

    @staticmethod
    def _random_split_(subject_list, train_size, rd):
        train_list, val_test_list = train_test_split(subject_list, train_size=train_size, random_state=rd)  ## 0.7/0.3
        # val_list, test_list = train_test_split(val_test_list, train_size=0.5, random_state=rd)  ## 0.15/0.15  TODO: TRAIN SIZE IS TO BE PUT BACK TO 0.5!!!
        val_list, test_list = val_test_list, val_test_list
        return train_list, val_list, test_list

    @staticmethod
    def transform_fit(transform, data_list, config):
        if transform is None:
            return transform
        if config is not None and config['DATA']['clinical_cols']:
            cols = config['DATA']['clinical_cols']
        else:
            cols = list(data_list.columns)
        for i, elem in enumerate(transform.transforms):
            if hasattr(elem, 'fit'):
                transform.transforms[i].fit(data_list.loc[:, cols])
        return transform

    def seed_worker(self, worker_id):
        np.random.seed(self.rd_worker)
        random.seed(self.rd_worker)
