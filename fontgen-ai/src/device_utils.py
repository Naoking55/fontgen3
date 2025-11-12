#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デバイス管理ユーティリティ - CUDA/MPS/CPU対応
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    最適なデバイスを取得

    優先順位:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU

    Args:
        use_gpu: GPUを使用するか
        gpu_id: GPU ID (CUDAの場合)

    Returns:
        torch.device
    """
    if not use_gpu:
        logger.info("Using CPU (GPU disabled)")
        return torch.device("cpu")

    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Device Count: {torch.cuda.device_count()}")
        return device

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
        logger.info(f"  PyTorch Version: {torch.__version__}")
        return device

    # CPU (フォールバック)
    logger.warning("GPU not available, using CPU")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    デバイス情報を取得

    Returns:
        Dict with device information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        })

    return info


def print_device_info():
    """デバイス情報を表示"""
    info = get_device_info()

    print("=" * 60)
    print(" Device Information")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")

    if info['cuda_available']:
        print(f"\nCUDA Information:")
        print(f"  CUDA Version: {info.get('cuda_version', 'N/A')}")
        print(f"  cuDNN Version: {info.get('cudnn_version', 'N/A')}")
        print(f"  Device Count: {info.get('device_count', 0)}")
        print(f"  Device Name: {info.get('device_name', 'N/A')}")

    if info['mps_available']:
        print(f"\nMPS (Apple Silicon) Available")

    print("=" * 60)


def optimize_for_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    デバイスに応じてモデルを最適化

    Args:
        model: PyTorchモデル
        device: デバイス

    Returns:
        最適化されたモデル
    """
    model = model.to(device)

    # CUDA最適化
    if device.type == "cuda":
        # cuDNN自動チューニング
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN auto-tuner")

    # MPS最適化
    elif device.type == "mps":
        # MPS特有の最適化があればここに追加
        logger.info("Model moved to MPS")

    return model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    バッチデータをデバイスに移動

    Args:
        batch: バッチデータ
        device: デバイス

    Returns:
        デバイスに移動したバッチ
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device, non_blocking=True)
        else:
            moved_batch[key] = value
    return moved_batch


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)

    print_device_info()

    device = get_device(use_gpu=True)
    print(f"\nSelected device: {device}")

    # テストテンソル
    x = torch.randn(4, 1, 128, 128)
    x = x.to(device)
    print(f"Tensor on device: {x.device}")
