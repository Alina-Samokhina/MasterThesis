import math

import torch
import torch.utils.data as data
import torchcde

from .activity import PersonData
from .p300 import P300Dataset


def get_data(num_timepoints=100):
    """Generates clockwise/counterclockwise spirals"""
    t = torch.linspace(0.0, 4 * math.pi, num_timepoints)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)

    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    return X, y


def load_dataset(
    ds="activity",
    timestamps=True,
    coeffs=False,
    batch_size=128,
    data_dir="../data/person",
):
    if ds == "activity":
        dataset = PersonData(data_dir=data_dir)
        train_ts = torch.Tensor(dataset.train_t)
        test_ts = torch.Tensor(dataset.test_t)
    elif ds == "p300":
        dataset = P300Dataset(data_dir=data_dir)
        dataset.get_data_for_experiments(True)
        train_ts = torch.Tensor(dataset.train_t)[:, :, None]
        test_ts = torch.Tensor(dataset.test_t)[:, :, None]
    else:
        raise ValueError(f'No such dataset: {ds}, try "activity"')

    train_x = torch.Tensor(dataset.train_x)
    test_x = torch.Tensor(dataset.test_x)

    train_y = torch.LongTensor(dataset.train_y)
    test_y = torch.LongTensor(dataset.test_y)

    print(train_ts.size(), train_x.size())

    if coeffs:
        train_x = torch.cat([train_ts, train_x], dim=2)
        train_x = torchcde.natural_cubic_coeffs(
            torch.Tensor(dataset.train_x)
        )  # , dtype=torch.float32))
        test_x = torch.cat([test_ts, test_x], dim=2)
        test_x = torchcde.natural_cubic_coeffs(
            torch.Tensor(dataset.test_x)
        )  # , dtype=torch.float32))

    if timestamps:
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = data.TensorDataset(test_x, test_ts, test_y)
    else:
        train = data.TensorDataset(train_x, train_y)
        test = data.TensorDataset(test_x, test_y)
    return_sequences = True

    counts = test_y.unique(return_counts=True)[1].to(torch.float)
    class_balance = counts / counts.min()

    trainloader = data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=4
    )
    in_features = train_x.size(-1)
    num_classes = int(torch.max(train_y).item() + 1)
    return (
        trainloader,
        testloader,
        in_features,
        num_classes,
        return_sequences,
        class_balance,
    )
