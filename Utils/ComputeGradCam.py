import os
os.system('pip install grad-cam')

import toml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
import SimpleITK as sitk
from Models.MixModel import MixModel
from DataGenerator.DataGenerator import DataModule
from Classification import transform_pipeline, build_model, load_model
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_random_seed(config):
    y = range(config['RUN']['bootstrap_n'])
    if 'random_state' in config['RUN'].keys():
        np.random.seed(seed=config['RUN']['random_state'])
    random_seed_list = np.random.randint(10000, size=len(y))
    for i in y:
        return random_seed_list[i]


def concat_vars(module_dict):
    records_out_shape = ([module_dict['records'].shape[0], module_dict['records'].shape[1]] +
                         list(module_dict['Image'].shape[2:]))
    expanded_records = module_dict['records'][:, :, None, None, None] * np.ones(records_out_shape, dtype=np.float32)
    concat = np.concatenate([expanded_records, module_dict['Image']], axis=1)
    return concat


def get_input_tensor(dataloader, mode='next'):
    if mode == 'next':
        batch = next(iter(dataloader.train_dataloader()))
        input_tensor = batch[0]  # Create an input tensor image for your model..
        input_tensor = concat_vars(input_tensor)
        # Note: input_tensor can be a batch tensor with several images!
    else:
        comb_batch = {'Image': [], 'records': [], 'slabel': []}
        for i in range(10):
            batch = dataloader.train_data[i]
            # sample_dict has keys 'Image', 'records', and 'slabel'
            comb_batch['Image'].append(batch[0]['Image'][None, :])
            comb_batch['records'].append(batch[0]['records'][None, :])
            comb_batch['slabel'].append(batch[0]['slabel'])
        batch = {'Image': np.concatenate(comb_batch['Image'], axis=0),
                 'records': np.concatenate(comb_batch['records'], axis=0),
                 'slabel': comb_batch['slabel']}
        input_tensor = concat_vars(batch)  # shape: (N, C_total, D, H, W)
    input_tensor = torch.from_numpy(input_tensor).to('cuda:0')
    return input_tensor, batch


class ImageWrapperModel(torch.nn.Module):
    def __init__(self, lightning_module, n_tab_vars):
        super().__init__()
        self.lightning_module = lightning_module
        self.n_tab_vars = n_tab_vars

    def forward(self, x_image):
        batch = {
            'Image': x_image[:, self.n_tab_vars:, :, :, :],
            'records': x_image[:, :self.n_tab_vars:, 0, 0, 0],
        }
        return self.lightning_module(batch)


def main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type='ckpt'):
    with open(config_path, 'r') as file:
        config_content = file.read()
        config_content = config_content[:config_content.index('Train transform')]
    config = toml.loads(config_content)
    random_seed = get_random_seed(config)
    seed_everything(random_seed, workers=True)
    clinical_cols = config['DATA']['clinical_cols']
    train_transform, val_transform = transform_pipeline(config)
    module_dict = build_model(config)
    model = MixModel(module_dict, config)
    subject_list = pd.read_csv(subject_list_path)
    split_list = pd.read_csv(split_list_path, skipfooter=1)
    dataloader = DataModule(
        subject_list,
        config=config,
        keys=config['MODALITY'].keys(),
        train_transform=train_transform,
        val_transform=val_transform,
        clinical_cols=clinical_cols,
        rd=np.int16(random_seed),
        rd_worker=np.int16(config['RUN']['random_state_dataloader']),
        inference=False,
        num_workers=10,
        prefetch_factor=5,
        split_list=split_list
    )

    ckpt_dict = {elem: list((Path(log_dir) / 'checkpoints').glob(f'*{dict_corr[elem]}'))[-1] for elem in dict_corr}
    model = load_model(ckpt_dict['lowest_val_loss'], log_dir, model, ckpt_type, config, module_dict)
    n_tab_vars = len(config['DATA']['given_clinical_cols'])
    wrapped_model = ImageWrapperModel(model, 5)
    wrapped_model.eval()

    target_layers = [list(model.module_dict['Image'].modules())[52]]
    input_tensor, batch = get_input_tensor(dataloader, 'next')
    # Note: input_tensor can be a batch tensor with several images!

    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(0)] * input_tensor.shape[0]

    # init cam
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # images: [B, C, H, W]

    for i, pat_arr in enumerate(grayscale_cam):
        print(i)
        gcam_sitk = sitk.GetImageFromArray(pat_arr.T)
        pat_ct = sitk.ReadImage(
            str(Path(subject_list.set_index('PatientID').loc[batch[0]['slabel'][i], 'CT_Path'])/'CT.nii.gz'))
        gcam_sitk.CopyInformation(pat_ct)
        out_path = Path(log_dir) / 'GradCams' / f"{batch[0]['slabel'][i]}.nii.gz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(gcam_sitk, str(out_path), True)

    # # Construct the CAM object once, and then re-use it on many images.
    # with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
    #     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     # In this example grayscale_cam has only one image in the batch:
    #     grayscale_cam = grayscale_cam[0, :]
    #     # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #     # # You can also get the model outputs without having to redo inference
    #     # model_outputs = cam.outputs

    return input_tensor, model, target_layers, grayscale_cam


if __name__ == "__main__":
    log_dir = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4"
    dict_corr = {'lowest_val_loss': 'lowest_val_loss.ckpt'}
    config_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4/Config.ini"
    subject_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/data_table.csv"
    split_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4/patient_list.csv"
    c_type = 'ckpt'  # 's_dict'
    results = main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type=c_type)

