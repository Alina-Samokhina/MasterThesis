from typing import Optional

import numpy as np
import torch
from torch import nn


class InsertDimention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, batch):
        size = batch.size()
        return batch.view(*size[: self.dim], 1, *size[self.dim :])


class EegNet(nn.Sequential, nn.Module):
    """EEG NEt implementation.
    https://arxiv.org/abs/1611.08024

    Input have to be shaped: (#batch, #channels, #time)
    """

    def __init__(
        self,
        input_size: tuple,
        rate: int = 50,
        F1: int = 4,
        D: int = 4,
        F2: Optional[int] = None,
        rate_decreases: tuple = (2, 4),
        dropout_rate: float = 0.25,
        classes: int = 2,
        *,
        device=None,
    ):
        if F2 is None:
            F2 = D * F1

        layers = [
            InsertDimention(1),
            # block 1
            nn.Conv2d(1, F1, (1, rate // 2), padding=(0, rate // 4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (input_size[0], 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, rate_decreases[0])),
            nn.Dropout(dropout_rate),
            # block 2
            nn.Conv2d(
                D * F1,
                D * F1,
                (1, rate // (2 * rate_decreases[0])),
                padding=(0, rate // (4 * rate_decreases[0])),
                groups=D * F1,
                bias=False,
            ),
            nn.Conv2d(D * F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, rate_decreases[1])),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(F2 * (input_size[1] // np.prod(rate_decreases)), classes),
        ]

        super().__init__(*layers)

        self.to(device, torch.float32)
