"""Example qualkit-action config: replace with your real models and data.

This synthetic example trains nothing; it builds a small classifier, makes a
4-bit-quantized copy, and serves random calibration data - enough to see the
gate run and fail/pass. Point the functions at your real artifacts.
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
_model = nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
_x = torch.randn(2000, 64)
_y = (_x[:, :10]).argmax(1)  # learnable synthetic labels

opt = torch.optim.Adam(_model.parameters(), lr=1e-2)
for _ in range(200):
    opt.zero_grad()
    nn.functional.cross_entropy(_model(_x), _y).backward()
    opt.step()

def get_reference_model():
    return copy.deepcopy(_model)

def get_deploy_model():
    m = copy.deepcopy(_model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data
                qm = 2 ** (4 - 1) - 1  # 4-bit
                sc = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qm
                mod.weight.data = torch.round(w / sc).clamp(-qm - 1, qm) * sc
    return m

def get_cal_loader():
    xv = torch.randn(1000, 64)
    yv = (xv[:, :10]).argmax(1)
    return DataLoader(TensorDataset(xv, yv), batch_size=256)

def get_probe_loader():
    xv = torch.randn(512, 64)
    yv = (xv[:, :10]).argmax(1)
    return DataLoader(TensorDataset(xv, yv), batch_size=256)
