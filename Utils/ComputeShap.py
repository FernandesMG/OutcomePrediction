import os
os.system('pip install shap')

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import toml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
import SimpleITK as sitk
import shap
import matplotlib.pyplot as plt
from Models.MixModel import MixModel
from DataGenerator.DataGenerator import DataModule
from Classification import transform_pipeline, build_model, load_model


device = "cuda:0"


def get_random_seed(config):
    y = range(config['RUN']['bootstrap_n'])
    if 'random_state' in config['RUN'].keys():
        np.random.seed(seed=config['RUN']['random_state'])
    random_seed_list = np.random.randint(10000, size=len(y))
    for i in y:
        return random_seed_list[i]


class TabImageWrapper(nn.Module):
    def __init__(self, lightning_module, target_idxs_to_explain=0):
        super().__init__()
        self.lightning_module = lightning_module
        self.target_idxs_to_explain = target_idxs_to_explain if type(target_idxs_to_explain) is list else [target_idxs_to_explain,]

    def forward(self, records, image):
        # records: [B, n_tab_vars]
        # image:   [B, C, D, H, W]
        batch = {
            "records": records,
            "Image": image,
        }
        y_hat = self.lightning_module(batch)  # e.g. [B] or [B, 1]

        # multi-target regression: pick one or keep them all, your choice
        # e.g. explain target 0:
        y_hat = y_hat[:, self.target_idxs_to_explain]            # (B, num_targets) -> (B, 1)

        return y_hat


def main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type='ckpt'):
    with open(config_path, 'r') as file:
        config_content = file.read()
        config_content = config_content[:config_content.index('Train transform')]
    config = toml.loads(config_content)
    config['MODEL']['batch_size'] = 60
    # config['MODEL']['tab_encoder'] = 'TabMLP'
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

    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    ckpt_dict = {elem: list((Path(log_dir) / 'checkpoints').glob(f'*{dict_corr[elem]}'))[-1] for elem in dict_corr}
    model = load_model(ckpt_dict['lowest_val_loss'], log_dir, model, ckpt_type, config, module_dict)
    model.eval().to(device)
    wrapped_model = TabImageWrapper(model).to(device)
    wrapped_model.eval()

    # ---- background data for SHAP (from training set) ----
    train_batch = next(iter(train_loader))
    train_x, train_y = train_batch[0], train_batch[1]

    background_records = train_x["records"][:config['MODEL']['batch_size']].to(device)  # (64, n_tab_vars)
    background_images = train_x["Image"][:config['MODEL']['batch_size']].to(device)  # (64, C, D, H, W)

    val_batch = next(iter(val_loader))
    val_x, val_y = val_batch[0], val_batch[1]

    records_to_explain = val_x["records"][:32].to(device)  # (32, n_tab_vars)
    images_to_explain = val_x["Image"][:32].to(device)  # (32, C, D, H, W)

    explainer = shap.DeepExplainer(
        wrapped_model,
        [background_records, background_images]
    )

    # For the selected validation samples:
    shap_values = explainer.shap_values(
        [records_to_explain, images_to_explain],
        check_additivity=False  # with False treat these explanations as approximate / qualitative rather than mathematically exact Shapley values. They’re still often very informative about which tabular features matter most.
    )

    shap_records = shap_values[0]  # SHAP for 'records'
    shap_images = shap_values[1]  # SHAP for 'Image' (ignore for now)

    # Each entry shap_records[i, j] = contribution of feature j (tabular variable) to the prediction for sample i, relative to the background distribution.

    # Turn SHAP values into feature importance
    clinical_cols = config['DATA']['clinical_cols']  # list of column names for 'records'
    # average absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_records), axis=0)[:, 0]  # (n_tab_vars, 1)[:, 0]

    # sort features by importance
    idx_sorted = np.argsort(-mean_abs_shap)
    for i in idx_sorted:
        print(f"{clinical_cols[i]}: {mean_abs_shap[i]:.4f}")

    # pretty plots
    records_np = records_to_explain.detach().cpu().numpy()

    shap.summary_plot(
        shap_records,
        records_np,
        feature_names=clinical_cols,
        max_display=20,  # top 20 features,
        show=False
    )

    # plt.tight_layout()
    # plt.savefig(Path(log_dir) / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(Path(log_dir) / 'shap_summary.png')
    plt.close()


if __name__ == "__main__":
    log_dir = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamRes/DSize1.0/fold4"
    dict_corr = {'lowest_val_loss': 'lowest_val_loss.ckpt'}
    config_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamRes/DSize1.0/fold4/Config.ini"
    subject_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/data_table.csv"
    split_list_path = "/home/ann/Code/OutcomePrediction/Logs/CT&Dose&Clinical/SurvivalPrediction/SimpleCNN/CoxLoss/AdamRes/DSize1.0/fold4/patient_list.csv"
    c_type = 'ckpt'  # 's_dict'
    main(log_dir, dict_corr, config_path, subject_list_path, split_list_path, ckpt_type=c_type)

