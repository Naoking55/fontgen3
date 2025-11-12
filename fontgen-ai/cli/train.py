#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習CLIスクリプト
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging
import torch
import random
import numpy as np

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import FontDataModule
from src.device_utils import get_device, optimize_for_device, print_device_info
from models.vae import FontVAE, VAELoss
from src.trainer import Trainer


def set_seed(seed: int):
    """再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="フォントVAE学習スクリプト")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイル (YAML)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="データディレクトリ",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="チェックポイントから再開",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="デバイス",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル",
    )

    args = parser.parse_args()

    # ログ設定
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # 設定読み込み
    config = load_config(args.config)
    logger.info(f"Loaded config from: {args.config}")

    # シード設定
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")

    # デバイス設定
    print_device_info()

    if args.device == "auto":
        device = get_device(use_gpu=True)
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # データセット
    logger.info("Loading dataset...")

    data_module = FontDataModule(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        image_size=config['model']['image_size'],
        augmentation=config['data']['augmentation']['enabled'],
        augmentation_params=config['data']['augmentation'],
    )

    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # モデル
    logger.info("Creating model...")

    model = FontVAE(
        image_size=config['model']['image_size'],
        z_content_dim=config['model']['z_content_dim'],
        z_style_dim=config['model']['z_style_dim'],
        input_channels=config['model'].get('image_channels', 1),
        output_channels=config['model'].get('image_channels', 1),
    )

    # モデルをデバイスに移動
    model = optimize_for_device(model, device)

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 損失関数
    criterion = VAELoss(
        recon_weight=config['training']['loss_weights']['reconstruction'],
        kl_weight=config['training']['loss_weights']['kl_divergence'],
        style_weight=config['training']['loss_weights']['style_consistency'],
        skeleton_weight=config['training']['loss_weights']['skeleton_consistency'],
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        **config['training']['optimizer_params']
    )

    # Scheduler
    scheduler = None
    if config['training'].get('scheduler') == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config['training']['scheduler_params']
        )

    # Resume
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        config={
            'gradient_clip': config['training'].get('gradient_clip', 1.0),
            'save_frequency': config['training']['checkpoint'].get('save_frequency', 5),
            'sample_frequency': config['logging']['save_samples'].get('frequency', 5),
            'early_stopping_patience': config['training']['early_stopping'].get('patience', 20),
        },
        scheduler=scheduler,
        use_tensorboard=config['logging']['tensorboard']['enabled'],
    )

    # 学習開始
    logger.info("Starting training...")

    results = trainer.train(num_epochs=config['training']['num_epochs'])

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
