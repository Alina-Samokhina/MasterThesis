# Based on the work of ODE LSTM authors Mathias Lechner ad Ramin Hasani
import numpy as np
import torch
import torch.utils.data as data
import torchcde

from .activity import PersonData
from .p300 import P300Dataset


# TO-DO reformat this. It works, but it's ugly.
def load_dataset(
    ds="activity",
    timestamps=True,
    coeffs=False,
    irregular=True,
    transpose=False,
    batch_size=128,
    data_dir="../data/person",
):
    """Obtains dataloaders for training diiferent networks on different datasets

    Args:
        ds: dataset to load. Options: activity/p300.
        timestamps: whether to have timestamps in dataloader.
            some architectures need it, some - don't.
        coeffs: whether to have features as raw data or its cubic pline coeffs.
            Needed for Neural CDE.
        irregular: whether to make the dataset irregular by dropping 20% of it's values.
        transpose: if False batch shape is (batch, seq_len, channels),
            if True -- (batch, channels, seq_len)
        batch_size: simply batch size.
        data_dir: directory, where data files are stored.
    """
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
        raise ValueError(f'No such dataset: {ds}, try "activity" or "p300"')

    train_x = torch.Tensor(dataset.train_x)
    test_x = torch.Tensor(dataset.test_x)

    train_y = torch.LongTensor(dataset.train_y)
    test_y = torch.LongTensor(dataset.test_y)

    # TODO make it function
    if irregular:
        seq_len_new = int(train_ts.size(1) * 0.8)
        new_ts = torch.zeros((train_ts.size(0), seq_len_new, 1))
        new_x = torch.zeros((train_x.size(0), seq_len_new, train_x.size(2)))
        new_y = torch.zeros((train_y.size(0), seq_len_new))
        for ts in range(len(train_ts)):
            irr_idc_train = np.random.choice(
                np.arange(0, train_ts.size(1)),
                size=int(0.8 * train_ts.size(1)),
                replace=False,
            )
            irr_idc_train.sort()
            new_ts[ts, :, :] = train_ts[ts, irr_idc_train, :]
            new_x[ts, :, :] = train_x[ts, irr_idc_train, :]
            if train_y.dim() > 1:
                new_y[ts, :] = train_y[ts, irr_idc_train]
        train_ts = new_ts.clone()
        train_x = new_x.clone()
        if train_y.dim() > 1:
            train_y = new_y.clone().to(torch.long)

        seq_len_new = int(test_ts.size(1) * 0.8)
        new_ts = torch.zeros((test_ts.size(0), seq_len_new, 1))
        new_x = torch.zeros((test_x.size(0), seq_len_new, test_x.size(2)))
        new_y = torch.zeros((test_y.size(0), seq_len_new))
        for ts in range(len(test_ts)):

            irr_idc_test = np.random.choice(
                np.arange(0, test_ts.size(1)),
                size=int(0.8 * test_ts.size(1)),
                replace=False,
            )
            irr_idc_test.sort()

            new_ts[ts, :, :] = test_ts[ts, irr_idc_test, :]
            new_x[ts, :, :] = test_x[ts, irr_idc_test, :]
            if test_y.dim() > 1:
                new_y[ts, :] = test_y[ts, irr_idc_test]

        test_ts = new_ts.clone()
        test_x = new_x.clone()
        if test_y.dim() > 1:
            test_y = new_y.clone().to(torch.long)

    # TODO make it function
    if transpose:
        xs = train_x.shape
        train_x = train_x.reshape(xs[0], xs[2], xs[1])
        xs = test_x.shape
        test_x = test_x.reshape(xs[0], xs[2], xs[1])

    in_features = train_x.size(-1)

    if coeffs:
        train_x = torch.cat([train_ts, train_x], dim=2)
        train_x = torchcde.natural_cubic_coeffs(torch.Tensor(dataset.train_x))
        test_x = torch.cat([test_ts, test_x], dim=2)
        test_x = torchcde.natural_cubic_coeffs(torch.Tensor(dataset.test_x))

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
    num_classes = int(torch.max(train_y).item() + 1)

    return (
        trainloader,
        testloader,
        in_features,
        num_classes,
        return_sequences,
        class_balance,
    )
