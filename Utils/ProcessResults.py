from fileinput import filename
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import chain
import monai
import torchvision
import SimpleITK as sitk
from pathlib import Path
from torch import nn


def inverse_transform_target(results, dataloader, config, include_binary=False):
    t_preproc = dataloader.target_preprocessing
    for i, col in enumerate(t_preproc):
        if not include_binary and config['MODEL']['modes'][i] == 'classification':
            continue
        results.loc[:, [f'Prediction_{col}']] = t_preproc[col].inverse_transform(results.loc[:, [f'Prediction_{col}']])
        results.loc[:, [f'Target_{col}']] = t_preproc[col].inverse_transform(results.loc[:, [f'Target_{col}']])
    return results


def get_results_table(results, dataloader, config):
    end_columns = ['Censored', 'Index'] if len(results[0]) == 4 else ['Index']
    pred_columns = [f'Prediction_{i}' for i in [config['DATA']['target']] + config['DATA']['additional_targets']]
    target_columns = [f'Target_{i}' for i in [config['DATA']['target']] + config['DATA']['additional_targets']]
    columns = pred_columns + target_columns + end_columns
    arr = [(np.array(list(chain(*[r[idx] for r in results])))) for idx in range(len(results[0]))]
    arr = np.concatenate([arr_[:, None] if arr_.ndim == 1 else arr_ for arr_ in arr], axis=1)
    tab = pd.DataFrame(arr, columns=columns)
    tab[config['DATA']['subject_label']] = dataloader.full_list.loc[tab['Index'], config['DATA']['subject_label']]
    tab['Train Set'] = tab[config['DATA']['subject_label']].isin(dataloader.train_list[config['DATA']['subject_label']])
    tab['Validation Set'] = tab[config['DATA']['subject_label']].isin(
        dataloader.val_list[config['DATA']['subject_label']])
    tab['Test Set'] = tab[config['DATA']['subject_label']].isin(dataloader.test_list[config['DATA']['subject_label']])
    tab = tab.set_index(config['DATA']['subject_label'], drop=True)
    return tab.drop('Index', axis=1)


def compute_reconstruction_performance(results, target, pat_ids, dataloader, config):
    subject_label = config['DATA']['subject_label']
    columns = [subject_label, 'MSE', 'MAE', 'Huber']
    mse_loss = nn.MSELoss(reduction='none')
    mae_loss = nn.L1Loss(reduction='none')
    huber_loss = nn.HuberLoss(reduction='none')
    arr = np.concatenate([mse_loss(results, target), mae_loss(results, target), huber_loss(results, target)],
                         axis=1).mean(axis=(2, 3, 4))
    arr = np.concatenate([np.array(pat_ids)[:, None], arr], axis=1)
    tab = pd.DataFrame(arr, columns=columns)
    tab['Train Set'] = tab[subject_label].isin(dataloader.train_list[subject_label])
    tab['Validation Set'] = tab[subject_label].isin(dataloader.val_list[subject_label])
    tab['Test Set'] = tab[subject_label].isin(dataloader.test_list[subject_label])
    tab = tab.set_index(subject_label, drop=True)
    return tab


def get_train_val_test_tab(dataloader, rd, config):
    tab = dataloader.full_list.loc[:, [config['DATA']['subject_label']]]
    tab['Train Set'] = tab[config['DATA']['subject_label']].isin(dataloader.train_list[config['DATA']['subject_label']])
    tab['Validation Set'] = tab[config['DATA']['subject_label']].isin(
        dataloader.val_list[config['DATA']['subject_label']])
    tab['Test Set'] = tab[config['DATA']['subject_label']].isin(dataloader.test_list[config['DATA']['subject_label']])
    tab = tab.set_index(config['DATA']['subject_label'], drop=True)
    tab.loc['random_seed', 'Train Set'] = rd
    return tab


def inverse_transform_dose_ct(results, dataloader, config):
    results = results.detach().clone()
    inverse_transform = get_inverse_transforms(dataloader)
    cts = None
    doses = None
    for i, target in enumerate(config['DATA']['reconstruction_target']):
        if target == 'CT':
            ct_idx = i
            cts = inverse_transform({'CT_target': results[:, i]})['CT_target']
            results[:, ct_idx] = cts
        if target == 'RTDOSE':
            dose_idx = i
            doses = inverse_transform({'RTDOSE_target': results[:, i]})['RTDOSE_target']
            results[:, dose_idx] = doses
    return results, cts, doses


def get_inverse_transforms(dataloader):
    for transform in dataloader.val_transform.transforms:
        if type(transform) is monai.transforms.ScaleIntensityRanged:
            if 'CT_target' in transform.keys:
                ct_new_a_min, ct_new_a_max, ct_new_b_min, ct_new_b_max = get_limits_from_transform_scaler(transform)
            if 'RTDOSE_target' in transform.keys:
                dose_new_a_min, dose_new_a_max, dose_new_b_min, dose_new_b_max = get_limits_from_transform_scaler(transform)
    inverse_transform = [
        monai.transforms.ScaleIntensityRanged(
            a_min=ct_new_a_min, a_max=ct_new_a_max, b_min=ct_new_b_min, b_max=ct_new_b_max, keys=['CT_target'],
            allow_missing_keys=True),
        monai.transforms.ScaleIntensityRanged(
            a_min=dose_new_a_min, a_max=dose_new_a_max, b_min=dose_new_b_min, b_max=dose_new_b_max,
            keys=['RTDOSE_target'], allow_missing_keys=True),
    ]
    inverse_transform = torchvision.transforms.Compose(inverse_transform)
    return inverse_transform


def get_limits_from_transform_scaler(transform):
    b_min = transform.scaler.b_min
    b_max = transform.scaler.b_max
    a_min = transform.scaler.a_min
    a_max = transform.scaler.a_max
    return b_min, b_max, a_min, a_max


def save_results_images_to_disk(images, pat_ids, image_type, out_dir, dataloader, reference_img_filename='CT'):
    subject_list = dataloader.full_list.set_index('PatientID')
    filename = 'Dose' if image_type == 'RTDOSE' else ('CT' if image_type == 'CT' else None)
    print(f'Saving prediction images to disk at {Path(out_dir)/"prediction_images"}...')
    for i in tqdm(range(images.shape[0])):
        image_path = subject_list.loc[pat_ids[i], f'{image_type}_Path']
        image_path = Path(image_path, f'{reference_img_filename}.nii.gz')
        original_image = sitk.ReadImage(image_path)
        prediction_image = sitk.GetImageFromArray(images[i].cpu().numpy().T)
        prediction_image.SetOrigin(original_image.GetOrigin())
        prediction_image.SetSpacing(original_image.GetSpacing())
        prediction_image.SetDirection(original_image.GetDirection())
        for meta in original_image.GetMetaDataKeys():
            prediction_image.SetMetaData(meta, original_image.GetMetaData(meta))
        pat_out_dir = Path(out_dir)/'prediction_images'/pat_ids[i]
        pat_out_dir.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(prediction_image, str(pat_out_dir / f'{filename}.nii.gz'), True)
