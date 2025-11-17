from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import torch
import copy
import pandas as pd
from torch import nn
import torch.distributed as dist
from pathlib import Path
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError, MeanSquaredError
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torcheval.metrics.aggregation.auc import AUC
from torcheval.metrics.toolkit import sync_and_compute
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR
from sksurv.metrics import concordance_index_censored
from monai.networks import nets
import inspect
import os
import csv
import json
from Losses import loss as losses
from Losses.loss import WeightedMSE, CrossEntropy, ConcordanceIndex, CoxPHLoss
from Utils.Diagnostics import (test_loss_computation, get_lr, get_label, is_global_zero, save_mid_slices_for_batch,
                               to_uint8_slice, slug)


def get_parameter_mean(model):
    weights = [p.data for p in model.parameters() if p.requires_grad]
    all_weights = torch.cat([w.flatten() for w in weights])
    mean_weight = all_weights.mean().item()
    return mean_weight


class MixModel(LightningModule):
    def __init__(self, module_dict, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.module_dict = module_dict
        self.config = config
        self.loss_fcns = self.get_loss_functions()
        self.activations = [getattr(torch.nn, elem)() for elem in self.config["MODEL"]["activations"]]
        loss_weights = config["MODEL"]["loss_weights"]
        self.loss_weights = torch.tensor(
            loss_weights if isinstance(loss_weights, list) else [loss_weights] * len(self.loss_fcns))
        self.loss_needs_events = [len(inspect.signature(f.forward).parameters) >= 3 for f in self.loss_fcns]
        pred_type = 'risk' if type(self.loss_fcns[0]) is CoxPHLoss else 'time'
        if config['MODEL']['backbone'] in ['efficientnet', 'simpleCNN']:
            self.head = nn.Identity()
        elif config['MODEL']['backbone'] == 'simpleCNNBackbone':
            self.head = nn.Sequential()
            fused_dim = config['MODEL']['backbone_out_c'] + config['MODEL']['tab_config'][-1]
            layers = ([fused_dim] + config['MODEL']['head_config'] + [config['DATA']['n_classes']])
            for i in range(len(layers) - 1):
                self.head += nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    nn.SiLU()
                ) if i + 1 < len(layers) - 1 else nn.Sequential(nn.Linear(layers[i], layers[i + 1]))  # if last layer, do not add nn.SiLu()
        else:
            layers = ([config['MODEL']['classifier_in']] + config['MODEL']['classifier_config'] +
                      [config['DATA']['n_classes']])
            self.head = nn.Sequential()
            for i in range(len(layers) - 1):
                self.head += nn.Sequential(
                    nn.Linear(layers[i], layers[i + 1]),
                    nn.Dropout(config['MODEL']['dropout_prob'])
                )
        self.head.apply(self.weights_init)
        self.survival_prediction_mode = config['MODEL']['modes'][0]
        self.val_dict = dict()
        self.train_dict = dict()

        if self.survival_prediction_mode == 'classification':
            self.train_accuracy = BinaryAccuracy()
            self.train_auc = BinaryAUROC()
            self.train_f1score = BinaryF1Score()
            self.validation_accuracy = BinaryAccuracy()
            self.validation_auc = BinaryAUROC()
            self.validation_f1score = BinaryF1Score()

        if self.survival_prediction_mode == 'regression':
            self.train_mae = MeanAbsoluteError()
            self.train_mape = MeanAbsolutePercentageError()
            self.train_c_index = ConcordanceIndex(pred_type)
            self.validation_mae = MeanAbsoluteError()
            self.validation_mape = MeanAbsolutePercentageError()
            self.validation_c_index = ConcordanceIndex(pred_type)

    def forward(self, data_dict):
        fused = torch.cat([self.module_dict[k](data_dict[k]) for k in self.module_dict if k in data_dict], dim=1)
        prediction = self.head(fused)
        return prediction

    def get_loss_functions(self):
        loss_fcns = list()
        for elem in self.config["MODEL"]["loss_functions"]:
            if hasattr(torch.nn, elem):
                loss_fcns.append(getattr(torch.nn, elem)(reduction="mean"))
            else:
                loss_fcns.append(getattr(losses, elem)(reduction="mean"))
        return loss_fcns

    def get_data_label_event(self, batch):
        if 'censor_label' in self.config['DATA']:
            data_dict, label, event = batch[:3]
        else:
            data_dict, label = batch
            event = torch.ones(label.shape, device=label.device)
        return data_dict, label, event

    def compute_loss_and_metrics(self, prediction: torch.Tensor,
                                 label: torch.Tensor,
                                 mode: str,
                                 stage: str,
                                 event: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mask = ~torch.isnan(label)
        loss = torch.tensor(0.0, device=prediction.device)
        for i in range(label.shape[1]):
            if mask[:, i].any():
                if self.loss_needs_events[i]:
                    loss += (self.loss_weights[i] *
                             self.loss_fcns[i](prediction[mask[:, i], i], label[mask[:, i], i], event[mask[:, i]]))
                else:
                    loss += (self.loss_weights[i] *
                             self.loss_fcns[i](prediction[mask[:, i], i], label[mask[:, i], i]))
        loss = loss / mask.any(dim=0).sum()
        survival_prediction = prediction[mask[:, 0], 0]
        survival_label = label[mask[:, 0], 0]
        survival_event = event[mask[:, 0]]  # ONLY IMPLEMENTED FOR CENSORING IN SURVIVAL TIME (1ST LABEL & PREDICTION)
        pred_final = self.activations[0](survival_prediction.detach())

        if survival_label.numel() > 0:
            if mode == 'classification':
                if stage == 'train':
                    self.train_accuracy(pred_final, survival_label)
                    self.train_auc(pred_final, survival_label)
                    self.train_f1score(pred_final, survival_label)
                elif stage == 'val':
                    self.validation_accuracy(pred_final, survival_label)
                    self.validation_auc(pred_final, survival_label)
                    self.validation_f1score(pred_final, survival_label)
            elif mode == 'regression':
                if stage == 'train':
                    self.train_mae(pred_final, survival_label)
                    self.train_mape(pred_final, survival_label)
                    self.train_c_index(pred_final, survival_label, survival_event)
                elif stage == 'val':
                    self.validation_mae(pred_final, survival_label)
                    self.validation_mape(pred_final, survival_label)
                    self.validation_c_index(pred_final, survival_label, survival_event)
        return loss, {'prediction': prediction, 'label': label}

    def training_step(self, batch, batch_idx):
        data_dict, label, event = self.get_data_label_event(batch)
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.survival_prediction_mode,
            stage='train' if self.training else 'val',
            event=event)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        # loss_v = test_loss_computation(prediction, label, event, lf=self.config["MODEL"]["loss_functions"])
        # self._log_high_loss_csv(
        #     prediction=prediction,
        #     loss=loss,
        #     loss_v=loss_v,
        #     event=event,
        #     data_dict=data_dict,
        #     label=label,
        #     batch_idx=batch_idx,
        #     csv_path=str(Path(self.logger.log_dir) / "high_loss_steps_train.csv"),
        #     specific_tag="",
        #     save_mid_slices=False,
        # )
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_train_epoch_end(self):
        if self.survival_prediction_mode == 'classification':
            self.log('train_accuracy_epoch', self.train_accuracy, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_auc_epoch", self.train_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('train_f1score_epoch', self.train_f1score, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
        elif self.survival_prediction_mode == 'regression':
            self.log('train_mae_epoch', self.train_mae, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_mape_epoch", self.train_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('train_c_index_epoch', self.train_c_index, on_step=False, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        data_dict, label, event = self.get_data_label_event(batch)
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.survival_prediction_mode,
            stage='train' if self.training else 'val',
            event=event)
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        # loss_v = test_loss_computation(prediction, label, event, lf=self.config["MODEL"]["loss_functions"])
        # self._log_high_loss_csv(
        #     prediction=prediction,
        #     loss=loss,
        #     loss_v=loss_v,
        #     event=event,
        #     data_dict=data_dict,
        #     label=label,
        #     batch_idx=batch_idx,
        #     csv_path=str(Path(self.logger.log_dir) / "high_loss_steps_val.csv"),
        #     save_mid_slices=False,
        # )
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_validation_epoch_end(self):
        if self.survival_prediction_mode == 'classification':
            self.log('validation_accuracy_epoch', self.validation_accuracy, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_auc_epoch', self.validation_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('validation_f1score_epoch', self.validation_f1score, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)
        elif self.survival_prediction_mode == 'regression':
            self.log('validation_mae_epoch', self.validation_mae, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_mape_epoch', self.validation_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('validation_c_index_epoch', self.validation_c_index, on_step=False, on_epoch=True,
                     sync_dist=True)

    def test_step(self, batch, batch_idx):
        data_dict, label, event = self.get_data_label_event(batch)
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.survival_prediction_mode,
            stage='test',
            event=event)
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch[0]
        prediction = self.forward(data_dict)
        prediction_final = torch.cat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(prediction.shape[1])], dim=1)
        return prediction_final, *batch[1:]

    def weights_init(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight.data)

    def weights_reset(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            m.reset_parameters()

    def configure_optimizers(self):
        def lr_lambda(epoch):
            warmup_epochs = self.config['MODEL']['lr_warmup_epochs']
            return (epoch + 1) / (warmup_epochs + 1) if epoch < warmup_epochs else 1.0

        opt_cls = getattr(torch.optim, self.config['MODEL']['optimizer'], torch.optim.Adam)
        opt_params = eval(self.config['MODEL']['opt_params'])
        optimizer = opt_cls(self.parameters(), lr=self.config['MODEL']['learning_rate'], **opt_params)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = getattr(lr_scheduler, self.config['MODEL']['lr_scheduler'], dict())
        scheduler_params = eval(self.config['MODEL']['lr_params'])
        decay_scheduler = scheduler(optimizer, **scheduler_params)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[self.config['MODEL']['lr_warmup_epochs']])
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}

######## ADDED FOR DIAGNOSTICS #########
    def _log_high_loss_csv(
            self,
            *,
            prediction,
            loss,
            loss_v,
            event,
            data_dict,
            label,
            batch_idx: int,
            csv_path: str = "high_loss_steps.csv",
            specific_tag: str = None,
            include_ct_slices_json: bool = False,
            include_dose_slices_json: bool = False,
            include_struct_slices_json: bool = False,
            save_mid_slices: bool = False,
    ):
        """Append one row with diagnostics for the current step (rank-0 only)."""
        if hasattr(self, "trainer") and not is_global_zero(self.trainer):
            return

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        # ---- Predictions (first column) ----
        pred_vec_t = prediction[:, 0].detach().float().cpu().reshape(-1)
        pred_vec = pred_vec_t.tolist()
        pred_mean = float(pred_vec_t.mean().item())
        pred_std = float(pred_vec_t.std(unbiased=False).item())
        pred_max_val = float(pred_vec_t.max().item())
        pred_min_val = float(pred_vec_t.min().item())
        pred_max_idx = int(pred_vec_t.argmax().item())
        pred_min_idx = int(pred_vec_t.argmin().item())

        slabels = data_dict.get("slabel", None)
        pred_max_label = get_label(slabels, pred_max_idx)
        pred_min_label = get_label(slabels, pred_min_idx)

        # ---- Loss scalar + per-sample losses ----
        loss_scalar = float(loss.detach().float().mean().cpu().item())
        loss_vec_t = loss_v.detach().float().cpu().reshape(-1)
        loss_vec = loss_vec_t.tolist()
        loss_mean = float(loss_vec_t.mean().item())
        loss_std = float(loss_vec_t.std(unbiased=False).item())
        loss_max_val = float(loss_vec_t.max().item())
        loss_max_idx = int(loss_vec_t.argmax().item())
        loss_max_label = get_label(slabels, loss_max_idx)

        # ---- Event stats ----
        event_t = event.detach().float().cpu().reshape(-1)
        event_mean = float(event_t.mean().item())
        event_std = float(event_t.std(unbiased=False).item())

        # ---- Weights stats ----
        with torch.no_grad():
            flat_w = torch.cat([p.detach().float().reshape(-1) for p in self.parameters()])
            weights_mean = float(flat_w.mean().item())
            weights_std = float(flat_w.std(unbiased=False).item())

        # ---- Learning rate & image mean ----
        lr = get_lr(self.trainer) if hasattr(self, "trainer") else None
        img = data_dict.get("Image", None)
        image_mean = float(img.detach().float().mean().cpu().item()) if torch.is_tensor(img) else None

        # ---- Labels + stats ----
        label_t = label.detach().float().cpu().reshape(-1)
        labels_list = label_t.tolist()
        label_mean = float(label_t.mean().item())
        label_std = float(label_t.std(unbiased=False).item())
        label_max_value = float(label_t.max().item())
        label_min_value = float(label_t.min().item())

        # ---- Epoch / step ----
        epoch = int(getattr(self, "current_epoch", -1))
        global_step = int(getattr(self, "global_step", -1))

        # ---- NEW: Save mid-slices (expects Image shape [B, 2, D1, D2, D3]) ----
        ct_slices_json = None
        dose_slices_json = None
        struct_slices_json = None
        if torch.is_tensor(img) and img.ndim == 5 and img.size(1) >= 2:
            # Directory to store images next to the CSV
            base_dir = os.path.dirname(csv_path) or "."
            slice_dir = os.path.join(base_dir, "slices")
            tag = f"ep{epoch}_gs{global_step}_b{batch_idx}" if specific_tag is None else specific_tag
            if save_mid_slices:
                ct_list, dose_list, struct_list = save_mid_slices_for_batch(img, slice_dir, tag, slabels)
                ct_slices_json = json.dumps(ct_list) if include_ct_slices_json else ""
                dose_slices_json = json.dumps(dose_list) if include_dose_slices_json else ""
                struct_slices_json = json.dumps(struct_list) if include_struct_slices_json and img.shape[1] == 3 else ""

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "batch_idx": int(batch_idx),

            # predictions & stats
            "predictions": json.dumps(pred_vec),
            "prediction_mean": pred_mean,
            "prediction_std": pred_std,
            "prediction_max_label": pred_max_label,
            "prediction_max_value": pred_max_val,
            "prediction_min_label": pred_min_label,
            "prediction_min_value": pred_min_val,

            # loss & stats
            "loss": loss_scalar,
            "losses": json.dumps(loss_vec),
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "loss_max_label": loss_max_label,
            "loss_max_value": loss_max_val,

            # event stats
            "event_mean": event_mean,
            "event_std": event_std,

            # weights stats
            "weights_mean": weights_mean,
            "weights_std": weights_std,

            # LR + image mean + labels
            "learning_rate": lr,
            "image_mean": image_mean,
            "labels": json.dumps(labels_list),
            "label_mean": label_mean,
            "label_std": label_std,
            "label_max_value": label_max_value,
            "label_min_value": label_min_value,
        }

        if save_mid_slices:  # NEW: file paths to the saved mid-slice JPEGs (JSON arrays)
            row["ct_mid_slices"] = ct_slices_json
            row["dose_mid_slices"] = dose_slices_json
            row["struct_mid_slices"] = struct_slices_json

        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

