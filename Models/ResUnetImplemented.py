from typing import Any, Dict, Tuple
import torch
import copy
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError, MeanSquaredError
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torcheval.metrics.aggregation.auc import AUC
from torcheval.metrics.toolkit import sync_and_compute
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR, CosineAnnealingLR
from pathlib import Path

from Losses import loss as losses
from Losses.loss import WeightedMSE, CrossEntropy, ConcordanceIndex, CoxPHLoss
from Utils.Diagnostics import test_loss_computation, get_lr, get_label, is_global_zero, save_mid_slices_for_batch
from Models.SimpleCNNs import ResUNet3D


class ResUNetImplemented(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.loss_fcn = self.get_loss_function()
        self.activation = getattr(torch.nn, self.config["MODEL"]["activations"])()
        self.model = ResUNet3D(config['DATA']['n_channel'], config['DATA']['n_classes'],
                               drop_path_max=config['MODEL']['dropout_prob'], anisotropic=True)
        self.model.apply(self.weights_init)
        self.prediction_mode = config['MODEL']['modes']
        self.val_dict = dict()
        self.train_dict = dict()

        if self.prediction_mode == 'classification':
            self.train_accuracy = BinaryAccuracy()
            self.train_auc = BinaryAUROC()
            self.train_f1score = BinaryF1Score()
            self.validation_accuracy = BinaryAccuracy()
            self.validation_auc = BinaryAUROC()
            self.validation_f1score = BinaryF1Score()

        if self.prediction_mode == 'regression':
            self.train_mae = MeanAbsoluteError()
            self.train_mape = MeanAbsolutePercentageError()
            # self.train_r2 = R2Score()
            self.validation_mae = MeanAbsoluteError()
            self.validation_mape = MeanAbsolutePercentageError()
            # self.validation_r2 = R2Score()

    def forward(self, data_dict):
        return self.model(data_dict['Image'])

    def get_loss_function(self):
        loss_func = self.config["MODEL"]["loss_function"]
        if hasattr(torch.nn, loss_func):
            return getattr(torch.nn, loss_func)(reduction="mean")
        else:
            return getattr(losses, loss_func)(reduction="mean")

    def compute_loss_and_metrics(self, prediction: torch.Tensor,
                                 label: torch.Tensor,
                                 mode: str,
                                 stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss = self.loss_fcn(prediction, label)

        if mode == 'classification':
            if stage == 'train':
                self.train_accuracy(self.activation(prediction), label)
                self.train_auc(self.activation(prediction), label)
                self.train_f1score(self.activation(prediction), label)
            elif stage == 'val':
                self.validation_accuracy(self.activation(prediction), label)
                self.validation_auc(self.activation(prediction), label)
                self.validation_f1score(self.activation(prediction), label)
        elif mode == 'regression':
            if stage == 'train':
                self.train_mae(self.activation(prediction), label)
                self.train_mape(self.activation(prediction), label)
            elif stage == 'val':
                self.validation_mae(self.activation(prediction), label)
                self.validation_mape(self.activation(prediction), label)
        return loss, {'prediction': prediction, 'label': label}

    def training_step(self, batch, batch_idx):
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.prediction_mode,
            stage='train')
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_train_epoch_end(self):
        if self.prediction_mode == 'classification':
            self.log('train_accuracy_epoch', self.train_accuracy, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_auc_epoch", self.train_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('train_f1score_epoch', self.train_f1score, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
        elif self.prediction_mode == 'regression':
            self.log('train_mae_epoch', self.train_mae, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_mape_epoch", self.train_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            # self.log('train_r2_epoch', self.train_r2, on_step=False, on_epoch=True, sync_dist=True,
            #          prog_bar=False)

    def validation_step(self, batch, batch_idx):
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.prediction_mode,
            stage='val')
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_validation_epoch_end(self):
        if self.prediction_mode == 'classification':
            self.log('validation_accuracy_epoch', self.validation_accuracy, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_auc_epoch', self.validation_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('validation_f1score_epoch', self.validation_f1score, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)
        elif self.prediction_mode == 'regression':
            self.log('validation_mae_epoch', self.validation_mae, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_mape_epoch', self.validation_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            # self.log('validation_r2_epoch', self.validation_r2, on_step=False, on_epoch=True,
            #          sync_dist=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(
            prediction,
            label,
            self.prediction_mode,
            stage='test')
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch[0]
        prediction = self.forward(data_dict)
        return prediction, *batch[1:]

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
        optimizer = opt_cls(self.parameters(), lr=self.config['MODEL']['learning_rate'])
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = getattr(lr_scheduler, self.config['MODEL']['lr_scheduler'])
        scheduler_params = eval(self.config['MODEL']['lr_params'])
        decay_scheduler = scheduler(optimizer, **scheduler_params)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[self.config['MODEL']['lr_warmup_epochs']])
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}
