import toml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
import SimpleITK as sitk
from Models.MixModel import MixModel
from DataGenerator.DataGenerator import DataModule
from Classification import transform_pipeline, build_model, infer_and_save_results, load_model
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


def main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type='ckpt'):
    config = toml.load(config_path)
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
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config['RUN']['max_epochs'],
        logger=False,
        benchmark=True
    )
    ckpt_dict = {elem: list((Path(log_dir) / 'checkpoints').glob(f'*{dict_corr[elem]}'))[-1] for elem in dict_corr}
    infer_and_save_results(ckpt_dict, log_dir, config, model, trainer, dataloader, ckpt_type, module_dict)


if __name__ == "__main__":
    # log_dir = "/home/dgs1/Software/Miguel/FedFLR/outcome-prediction-test/outcome_prediction_test/Logs/CT&Target&Dose/SurvivalPrediction/EfficientNetB0/DGS1RUMC2ChLongLR_v2/fold0"
    # log_dir = "/home/dgs1/Software/Miguel/FedFLR/outcome-prediction-test/outcome_prediction_test/Logs/CT&Target&Dose/SurvivalPrediction/EfficientNetB0/DGS1RUMC2ChLongLR_0.4/fold0"
    log_dir = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4"
    # dict_corr = {'lowest_val_loss': 'lowest_val_loss.ckpt', 'best_main_target': 'best_main_target.ckpt'}
    # dict_corr = {'lowest_val_loss': 'best_model_round_152.ckpt'}
    # dict_corr = {'latest_model': 'latest_model.ckpt'}
    dict_corr = {'lowest_val_loss': 'lowest_val_loss.ckpt'}
    # config_path = "/home/dgs1/Software/Miguel/FedFLR/outcome-prediction-test/outcome_prediction_test/OPConfigurationInference.ini"
    config_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4/Config.ini"
    # subject_list_path = "/home/dgs1/Software/Miguel/FedFLR/outcome-prediction-test/outcome_prediction_test/Logs/CT&Target&Dose/SurvivalPrediction/EfficientNetB0/DGS1RUMC2ChLongLR_0.4/fold0/data_table.csv"
    subject_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/data_table.csv"
    # split_list_path = "/home/dgs1/Software/Miguel/FedFLR/outcome-prediction-test/outcome_prediction_test/Logs/CT&Target&Dose/SurvivalPrediction/EfficientNetB0/DGS1RUMC2ChLongLR_0.4/fold0/checkpoints/patient_list_dgs1.csv"
    split_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamResContOnly/DSize1.0/fold0_tab16.8_head16_LR1e-4/patient_list.csv"
    c_type = 'ckpt'  # 's_dict'
    main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type=c_type)

