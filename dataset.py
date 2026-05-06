# -*- coding: utf-8 -*-
"""
CSI dataset utilities: loading, preprocessing, augmentation.
"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


# ======================== Dataset ========================

class CSIDataset(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray):
        self.data  = data.astype(np.float32, copy=False)   # [N,2,H,W]
        self.label = label.astype(np.float32, copy=False)  # [N,2]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i]), torch.from_numpy(self.label[i])


class GPUCachedDataset(Dataset):
    """Wrap a CSIDataset by moving all data to GPU once.
    Eliminates per-batch CPU→GPU transfer at the cost of GPU memory."""
    def __init__(self, dataset: CSIDataset, device: torch.device):
        self.data  = torch.from_numpy(dataset.data).to(device)
        self.label = torch.from_numpy(dataset.label).to(device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        return self.data[i], self.label[i]


# ======================== Build & Load ========================

def build_datasets_cost2100(data_dir: str, envir: str = "indoor"):
    if envir == "indoor":
        f_tr, f_va, f_te = "DATA_Htrainin.mat", "DATA_Hvalin.mat", "DATA_Htestin.mat"
    else:
        f_tr, f_va, f_te = "DATA_Htrainout.mat", "DATA_Hvalout.mat", "DATA_Htestout.mat"

    def _load_one(fname):
        mat = sio.loadmat(os.path.join(data_dir, fname))
        x = mat["HT"].astype(np.float32)          # [N, D]
        N, D = x.shape
        hw = int(round(np.sqrt(D / 2)))
        assert 2 * hw * hw == D, f"HT dim {D} not equal to 2*H*W"
        x = x.reshape(N, 2, hw, hw)

        y = np.zeros((N, 2), dtype=np.float32)
        y[:, 0] = np.arange(N, dtype=np.float32)
        y[:, 1] = 1.0
        return x, y

    x_tr, y_tr = _load_one(f_tr)
    x_va, y_va = _load_one(f_va)
    x_te, y_te = _load_one(f_te)
    return CSIDataset(x_tr, y_tr), CSIDataset(x_va, y_va), CSIDataset(x_te, y_te)


def build_loaders(train_set, val_set, test_set, batch_size,
                  num_workers=0, drop_last=True, gpu_preload=False,
                  device=None):
    if gpu_preload:
        assert device is not None, "gpu_preload requires a device"
        train_set = GPUCachedDataset(train_set, device)
        val_set   = GPUCachedDataset(val_set,   device)
        test_set  = GPUCachedDataset(test_set,  device)
        # CUDA tensors cannot be shared across DataLoader workers
        num_workers = 0

    kw = dict(num_workers=num_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last, **kw)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              drop_last=False, **kw)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              drop_last=False, **kw)
    return train_loader, val_loader, test_loader


def load_data(file_path, batch_size, num_workers=0, drop_last=True,
              dataset_name="cost2100", envir="indoor",
              gpu_preload=False, device=None):
    if dataset_name not in ("cost2100", "csinet"):
        raise ValueError(f"Only supports COST2100, got dataset_name={dataset_name}")
    train_set, val_set, test_set = build_datasets_cost2100(file_path, envir=envir)
    return build_loaders(train_set, val_set, test_set, batch_size,
                         num_workers=num_workers, drop_last=drop_last,
                         gpu_preload=gpu_preload, device=device)


# ======================== Preprocessing ========================

# Models that use centred-and-scaled preprocessing: (x - 0.5) * scale_factor
_SNN_MODEL_NAMES = {"spikingcsinet", "spikeconvcsinet", "snnconvenccsinet", "snnclnetenccsinet"}


def data_preprocess(x: torch.Tensor, model_name: str, scale_factor: float):
    """
    SNN models: (x - 0.5) * scale_factor
    ANN (csinet):  identity
    """
    if model_name in _SNN_MODEL_NAMES:
        return (x - 0.5) * float(scale_factor)
    return x


def restore_channel(yhat: torch.Tensor, model_name: str, scale_factor: float):
    """
    Restore network output to the centred channel domain (≈ x - 0.5):
    SNN models: yhat / scale_factor
    ANN:        yhat - 0.5
    """
    if model_name in _SNN_MODEL_NAMES:
        return yhat / float(scale_factor)
    return yhat - 0.5


# ======================== Augmentation ========================

def aug_csi(x, phase_rotate=True, noise_std=0.0, phase_bins: int = 0):
    """
    CSI augmentation on x=[B,2,H,W].
    phase_bins: 0 = continuous uniform, N > 0 = discrete N-phase.
    """
    assert x.dim() == 4 and x.size(1) == 2
    B = x.size(0)

    if phase_rotate:
        if phase_bins > 0:
            k   = torch.randint(0, phase_bins, (B,), device=x.device)
            phi = (2.0 * torch.pi / phase_bins) * k.to(x.dtype)
        else:
            phi = 2.0 * torch.pi * torch.rand(B, device=x.device, dtype=x.dtype)

        c = phi.cos().view(B, 1, 1, 1)
        s = phi.sin().view(B, 1, 1, 1)
        re, im = x[:, 0:1], x[:, 1:2]
        x = torch.cat([re * c - im * s, re * s + im * c], dim=1)

    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x
