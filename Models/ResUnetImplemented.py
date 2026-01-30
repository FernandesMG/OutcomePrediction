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
from torchvision.utils import save_image

from Losses import loss as losses
from Losses.loss import WeightedMSE, CrossEntropy, ConcordanceIndex, CoxPHLoss
from Utils.Diagnostics import test_loss_computation, get_lr, get_label, is_global_zero, save_mid_slices_for_batch
from Models import SimpleCNNs
from Models.SimpleCNNs import ResUNet3D, ResUNet3DWLateFusion


class ResUNetImplemented(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.loss_fcn = self.get_loss_function()
        activation = getattr(torch.nn, self.config["MODEL"]["activations"])
        activation_params = eval(config['MODEL']['activation_params'])
        self.activation = activation(**activation_params)
        model = getattr(SimpleCNNs, config['MODEL']['backbone'])
        model_params = eval(config['MODEL']['backbone_params'])
        self.model = model(config['DATA']['n_channel'], config['DATA']['n_classes'],
                           drop_path_max=config['MODEL']['dropout_prob'], anisotropic=True, **model_params)
        self.model.apply(self.weights_init)
        self.prediction_mode = config['MODEL']['modes']
        self.val_dict = dict()
        self.train_dict = dict()
        self._saved_tr_example_this_epoch = False
        self._saved_val_example_this_epoch = False

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
        if self.config['RECORDS']['records']:
            return self.activation(self.model(data_dict['Image'], data_dict['records']))
        else:
            return self.activation(self.model(data_dict['Image']))

    def get_loss_function(self):
        loss_func = self.config["MODEL"]["loss_function"]
        if hasattr(torch.nn, loss_func):
            return getattr(torch.nn, loss_func)(reduction='none')
        else:
            return getattr(losses, loss_func)(reduction='none')

    def compute_loss_and_metrics(self, prediction: torch.Tensor,
                                 label: torch.Tensor,
                                 mode: str,
                                 stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss = self.loss_fcn(prediction, label)
        N = loss.shape[-3] * loss.shape[-2] * loss.shape[-1] # number of voxels inside volume
        k = int(self.config['MODEL']['top_k_perc']/100 * N)
        top_k_loss, _ = torch.topk(loss.flatten(start_dim=-3, end_dim=-1), k, sorted=False) # keep only the top k largest values in loss
        loss = top_k_loss.mean()

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
        if ((not self._saved_tr_example_this_epoch)
                and is_global_zero(self.trainer)
                and (self.current_epoch < 20 or self.current_epoch % 10 == 0)):
            self._save_random_mid_slice(
                images=data_dict['Image'],
                outputs=prediction,
                pat_ids=data_dict['slabel'],
                mode='train'
            )
            self._saved_tr_example_this_epoch = True
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
        if ((not self._saved_val_example_this_epoch)
                and is_global_zero(self.trainer)
                and (self.current_epoch < 20 or self.current_epoch % 10 == 0)):
            self._save_random_mid_slice(
                images=data_dict['Image'],
                outputs=prediction,
                pat_ids=data_dict['slabel'],
                mode='validation',
            )
            self._saved_val_example_this_epoch = True
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
        return prediction, *batch[1:], batch[0]['slabel']

    def on_train_epoch_start(self) -> None:
        # reset at the beginning of each training epoch
        self._saved_tr_example_this_epoch = False

    def on_validation_epoch_start(self) -> None:
        # reset at the beginning of each validation epoch
        self._saved_val_example_this_epoch = False

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

    def _save_random_mid_slice(self, images: torch.Tensor, outputs: torch.Tensor, pat_ids = None, mode = None) -> None:
        """
        Save mid-slice of input volume (150x100x35) and corresponding model output
        for a single random instance in the current batch.
        Assumes shapes:
            images:  [B, C, *, *, *]
            outputs: [B, C_out, *, *, *]
        where one of the spatial dims is 35 (depth).
        """

        # Detach to CPU
        images = images.detach().cpu()
        outputs = outputs.detach().cpu()

        # Pick random patient in batch
        batch_size = images.shape[0]
        if batch_size == 0:
            return
        idx = torch.randint(0, batch_size, (1,)).item()

        # Take first channel for visualization
        vol_in = images[idx, 0]  # shape: (S1, S2, S3)
        vol_out = outputs[idx, 0]  # shape: (S1, S2, S3) or similar

        # Find which axis is the depth (size == 35)
        in_shape = vol_in.shape
        depth_axis = -1

        # Mid index along depth
        mid_idx = in_shape[depth_axis] // 2

        # Get 2D mid-slices (H x W)
        slice_in = vol_in.select(dim=depth_axis, index=mid_idx)
        slice_out = vol_out.select(dim=depth_axis, index=mid_idx)

        # Ensure shapes are [1, H, W] for save_image
        slice_in = slice_in.unsqueeze(0)
        # Apply activation to output for nicer visualization (e.g. sigmoid)
        slice_out = self.activation(slice_out).unsqueeze(0)

        # Build output directory: <log_dir>/mid_slices
        if self.logger is not None and hasattr(self.logger, "log_dir"):
            base_dir = Path(self.logger.log_dir)
        else:
            base_dir = Path(".")

        out_dir = base_dir / "mid_slices"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filenames (epoch + random index)
        epoch = self.current_epoch
        patient = idx if pat_ids is None else pat_ids[idx]
        mode = '' if mode is None else ('tr' if mode == 'train' else 'val')
        in_path = out_dir / f"{mode}_epoch_{epoch:03d}_patient_{patient}_input.png"
        out_path = out_dir / f"{mode}_epoch_{epoch:03d}_patient_{patient}_output.png"

        save_image(slice_in, str(in_path))
        save_image(slice_out, str(out_path))

