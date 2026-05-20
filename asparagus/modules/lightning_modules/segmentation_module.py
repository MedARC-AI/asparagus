import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional
import wandb
from asparagus.functional.metrics.utils import format_multilabel_metrics
from asparagus.functional.reverse_preprocessing import reverse_preprocessing
from asparagus.modules.lightning_modules.base_module import BaseModule
from gardening_tools.functional.metrics import (
    FN,
    FP,
    TP,
    dice,
    f1,
    jaccard,
    precision,
    sensitivity,
    specificity,
    total_pos_gt,
    total_pos_pred,
    volume_similarity,
)
from gardening_tools.functional.paths.write import save_json
from gardening_tools.functional.transforms.cropping_and_padding import (
    fit_patch_size_to_image_size,
)
from gardening_tools.modules.losses.deep_supervision import DeepSupervisionLoss
from gardening_tools.modules.losses.DiceCE import DiceCE
from gardening_tools.modules.metrics import GeneralizedDiceScore
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score
from torchvision import transforms
from typing import Optional


class SegmentationModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-2,
        warmup_epochs: int = 10,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        deep_supervision: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        optimizer: str = "SGD",
        inference_patch_size: list = [],
        inference_mode: str = "3D",
        test_output_path: str = None,
        log_image_every_n_epochs: int = 50,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        load_decoder: bool = True,
        repeat_stem_weights: bool = True,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            decoder_warmup_epochs=decoder_warmup_epochs,
            cosine_period_ratio=cosine_period_ratio,
            compile_mode=compile_mode,
            weights=weights,
            optimizer=optimizer,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            weight_decay=weight_decay,
            nesterov=nesterov,
            momentum=momentum,
            load_decoder=load_decoder,
            repeat_stem_weights=repeat_stem_weights,
        )
        self.inference_mode = inference_mode
        self.inference_patch_size = inference_patch_size
        self.test_output_path = test_output_path
        self.num_classes = model.num_classes
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.deep_supervision = deep_supervision

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")

        self.train_loss = DiceCE()
        self.val_loss = DiceCE()

        if self.deep_supervision:
            self.train_loss = DeepSupervisionLoss(loss=self.train_loss, weights=None)

    @staticmethod
    def _pad_image_for_sliding_window_inference(x, patch_size):
        spatial_shape = tuple(x.shape[2:])
        if len(spatial_shape) != len(patch_size):
            raise ValueError(
                f"Image spatial shape {spatial_shape} is incompatible with inference patch size {patch_size}."
            )

        pad_widths = []
        for dim_size, min_size in zip(spatial_shape, patch_size):
            pad_total = max(0, int(min_size) - int(dim_size))
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_widths.append((pad_before, pad_after))

        if not any(before or after for before, after in pad_widths):
            return x, spatial_shape, pad_widths

        # This is temporary inference padding only. Do not write it into
        # properties["pad_box"], which describes offline asp_process padding.
        pad_args = []
        for pad_before, pad_after in reversed(pad_widths):
            pad_args.extend([pad_before, pad_after])
        return F.pad(x, pad_args, mode="constant", value=float(x.min())), spatial_shape, pad_widths

    @staticmethod
    def _crop_logits_to_spatial_shape(logits, spatial_shape, pad_widths):
        if not any(before or after for before, after in pad_widths):
            return logits

        slices = [slice(None), slice(None)]
        for dim_size, (pad_before, _) in zip(spatial_shape, pad_widths):
            slices.append(slice(pad_before, pad_before + dim_size))
        return logits[tuple(slices)]

    def _sliding_window_predict_with_inference_padding(self, x):
        x, spatial_shape, pad_widths = self._pad_image_for_sliding_window_inference(
            x=x,
            patch_size=self.inference_patch_size,
        )
        logits = self.model.sliding_window_predict(
            data=x,
            # The old fitted patch size can become [64, 64, 0] for shallow
            # Task 2 volumes such as [512, 512, 21], so we pad the image and
            # keep the configured inference window instead.
            # patch_size=fit_patch_size_to_image_size(self.inference_patch_size, list(x.shape[2:])),
            patch_size=self.inference_patch_size,
            overlap=0.5,
        )

        # Crop away transient inference padding before reverse_preprocessing;
        # reverse_preprocessing expects logits in the asp_process tensor shape.
        return self._crop_logits_to_spatial_shape(logits, spatial_shape, pad_widths)

    def configure_metrics(self, prefix: str):
        return MetricCollection(
            {
                f"{prefix}/dice": GeneralizedDiceScore(
                    num_classes=self.num_classes,
                    weight_type="linear",
                    per_class=True,
                    input_format="index",
                ),
                f"{prefix}/F1": MulticlassF1Score(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average=None,
                ),
            },
        )

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.train_loss(pred, y)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of
            # (downsampled) tensors. We only need the original ground truth and
            # its corresponding prediction which is always the first entry in each list.
            pred = pred[0]
            y = y[0]

        metrics = self.train_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="train",
                task_type="segmentation",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        pred = self.model(x)
        loss = self.val_loss(pred, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        metrics = self.val_metrics(pred, y.squeeze(1))
        self.log_dict(
            format_multilabel_metrics(metrics, ignore_index=self.ignore_index_in_metrics),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if (
            self.current_epoch > 0
            and batch_idx == 0
            and wandb.run is not None
            and self.current_epoch % self.log_image_every_n_epochs == 0
        ):
            self._log_dict_of_images_to_wandb(
                {
                    "input": x.detach().cpu().to(torch.float32).numpy(),
                    "target": y.detach().cpu().to(torch.float32).numpy(),
                    "output": pred.detach().cpu().to(torch.float32).numpy(),
                    "file": batch["file_path"],
                },
                log_key="val",
                task_type="segmentation",
            )

    def on_test_epoch_start(self):
        self.test_metrics = [
            dice,
            f1,
            jaccard,
            precision,
            sensitivity,
            specificity,
            TP,
            FP,
            FN,
            total_pos_gt,
            total_pos_pred,
            volume_similarity,
        ]
        self.results = {}
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x = batch["image"]

        # # Old
        # logits = self.model.sliding_window_predict(
        #     data=x,
        #     patch_size=fit_patch_size_to_image_size(self.inference_patch_size, list(x.shape[2:])),
        #     overlap=0.5,
        # )
        # Replaced with a version that correctly patches
        logits = self._sliding_window_predict_with_inference_padding(x)

        src_logits = reverse_preprocessing(logits, batch["properties"])
        src_label = batch["src_label"]
        self.results[batch["file_path"]] = self.compute_metrics_from_confusion_matrix(src_logits, src_label)

    def on_test_epoch_end(self):
        avg_results = {}
        first_file = list(self.results.keys())[0]
        logging.info(f"Test results for {len(self.results)} files:")
        for label in self.results[first_file].keys():
            avg_results[label] = {}
            for metric in self.results[first_file][label].keys():
                avg_results[label][metric] = round(
                    np.nanmean([self.results[path][label][metric] for path in self.results]),
                    4,
                )
                logging.info(f"{label} {metric}: {avg_results[label][metric]}")
        self.results["mean"] = avg_results
        os.makedirs(os.path.split(self.test_output_path)[0], exist_ok=True)
        save_json(self.results, self.test_output_path)

    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        # # Old
        # logits = self.model.sliding_window_predict(
        #     data=x,
        #     patch_size=self.inference_patch_size,
        #     overlap=0.5,
        # )
        # Replaced with a version that correctly patches
        logits = self._sliding_window_predict_with_inference_padding(x)
        logits = reverse_preprocessing(
            array=logits,
            image_properties=batch["properties"],
        )
        batch["logits"] = logits
        return batch

    def compute_metrics_from_confusion_matrix(self, logits, label):
        metrics = {}
        labels = logits.shape[1]
        cmat = torchmetrics.functional.confusion_matrix(
            logits, label.squeeze(1), task="multiclass", num_classes=logits.shape[1]
        )
        for label in range(labels):
            metrics_for_label = {}
            tp = cmat[label, label]
            fp = torch.sum(cmat[:, label]) - tp
            fn = torch.sum(cmat[label, :]) - tp
            tn = torch.sum(cmat) - tp - fp - fn
            for metric in self.test_metrics:
                metrics_for_label[metric.__name__] = float(metric(tp, fp, tn, fn))
            metrics[str(label)] = metrics_for_label
        return metrics
