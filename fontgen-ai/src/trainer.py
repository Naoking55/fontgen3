#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer - 学習ループの管理
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional
import time

from .metrics import calculate_all_metrics
from .device_utils import move_batch_to_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    VAE学習マネージャー
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        config: Dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_tensorboard: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.config = config
        self.scheduler = scheduler
        self.use_tensorboard = use_tensorboard

        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        # TensorBoard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        # 学習状態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 学習履歴
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> Dict[str, float]:
        """1エポックの学習"""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'style_loss': 0.0,
            'skeleton_loss': 0.0,
        }
        epoch_metrics = {
            'mse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # データをデバイスに移動
            batch = move_batch_to_device(batch, self.device)

            images = batch['image']
            font_ids = batch['font_id']
            char_ids = batch['char_id']

            # Forward
            outputs = self.model(images)
            losses = self.criterion(outputs, images, font_ids, char_ids)

            # Backward
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )

            self.optimizer.step()

            # 損失を記録
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            # メトリクスを計算
            with torch.no_grad():
                metrics = calculate_all_metrics(outputs['reconstructed'], images)
                for key in epoch_metrics.keys():
                    epoch_metrics[key] += metrics[key]

            # TensorBoard
            if self.use_tensorboard and self.global_step % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f"Train/{key}", value.item(), self.global_step)

            # プログレスバー更新
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item(),
                'ssim': metrics['ssim'],
            })

            self.global_step += 1

        # エポック平均
        num_batches = len(self.train_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches

        return {**epoch_losses, **epoch_metrics}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """検証"""
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'style_loss': 0.0,
            'skeleton_loss': 0.0,
        }
        epoch_metrics = {
            'mse': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
        }

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # データをデバイスに移動
            batch = move_batch_to_device(batch, self.device)

            images = batch['image']
            font_ids = batch['font_id']
            char_ids = batch['char_id']

            # Forward
            outputs = self.model(images)
            losses = self.criterion(outputs, images, font_ids, char_ids)

            # 損失を記録
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            # メトリクスを計算
            metrics = calculate_all_metrics(outputs['reconstructed'], images)
            for key in epoch_metrics.keys():
                epoch_metrics[key] += metrics[key]

            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'ssim': metrics['ssim'],
            })

        # エポック平均
        num_batches = len(self.val_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches

        return {**epoch_losses, **epoch_metrics}

    def save_checkpoint(self, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 最新のチェックポイント
        checkpoint_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # ベストモデル
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

        # エポックごと
        if (self.current_epoch + 1) % self.config.get('save_frequency', 5) == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1:03d}.pth"
            torch.save(checkpoint, epoch_path)

    def save_samples(self, num_samples: int = 16):
        """サンプル画像を保存"""
        self.model.eval()

        with torch.no_grad():
            # バッチを取得
            batch = next(iter(self.val_loader))
            batch = move_batch_to_device(batch, self.device)

            images = batch['image'][:num_samples]
            outputs = self.model(images)
            reconstructed = outputs['reconstructed'][:num_samples]

            # 画像を結合
            import torchvision.utils as vutils

            comparison = torch.cat([images, reconstructed], dim=0)
            grid = vutils.make_grid(comparison, nrow=num_samples, normalize=True)

            # 保存
            save_path = self.samples_dir / f"epoch_{self.current_epoch + 1:03d}.png"
            vutils.save_image(grid, save_path)
            logger.info(f"Saved samples: {save_path}")

            # TensorBoard
            if self.use_tensorboard:
                self.writer.add_image(
                    "Reconstruction",
                    grid,
                    self.current_epoch
                )

    def train(self, num_epochs: int):
        """学習実行"""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 学習
            train_results = self.train_epoch()
            self.train_losses.append(train_results['total_loss'])

            # 検証
            val_results = self.validate()
            self.val_losses.append(val_results['total_loss'])

            # ログ
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_results['total_loss']:.4f}")
            logger.info(f"    - Recon: {train_results['recon_loss']:.4f}")
            logger.info(f"    - KL: {train_results['kl_loss']:.4f}")
            logger.info(f"    - SSIM: {train_results['ssim']:.4f}")
            logger.info(f"  Val Loss: {val_results['total_loss']:.4f}")
            logger.info(f"    - SSIM: {val_results['ssim']:.4f}")

            # TensorBoard
            if self.use_tensorboard:
                for key, value in train_results.items():
                    self.writer.add_scalar(f"Epoch/Train_{key}", value, epoch)
                for key, value in val_results.items():
                    self.writer.add_scalar(f"Epoch/Val_{key}", value, epoch)

                # Learning rate
                if self.optimizer.param_groups:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("Learning_Rate", lr, epoch)

            # ベストモデルチェック
            val_loss = val_results['total_loss']
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # チェックポイント保存
            self.save_checkpoint(is_best=is_best)

            # サンプル画像保存
            if (epoch + 1) % self.config.get('sample_frequency', 5) == 0:
                self.save_samples()

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            early_stop_patience = self.config.get('early_stopping_patience', 20)
            if self.patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # 学習終了
        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.use_tensorboard:
            self.writer.close()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
