import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import make_eeg_pipe, slice_epochs


class P300Dataset(Dataset):

    ch_names = ["Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2"]
    sampling_rate = 500.0
    url = "https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip"  # noqa: E501

    _hdf_path = "p300dataset"

    _act_dtype = np.dtype(
        [
            ("id", np.int),
            ("target", np.int),
            ("is_train", np.bool),
            ("prediction", np.int),
            ("sessions", np.object),  # list of `_session_dtype`
        ]
    )
    _session_dtype = np.dtype(
        [("eeg", np.object), ("starts", np.object), ("stimuli", np.object)]
    )

    def __init__(self, data_dir=None, get_data=False):
        self.eeg = []
        self.labels = []

        self.data_dir = data_dir
        self.markup = pd.read_csv(f"{self.data_dir}/meta.csv")
        self.decimation = 10
        self.rate = int(self.sampling_rate / self.decimation)
        self.transformation = make_eeg_pipe(
            self.sampling_rate, self.decimation, 0.1, 15, 35
        )
        if get_data:
            self.data = self.get_data()
        else:
            self.data = []

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def read_hdf(cls, filename) -> np.ndarray:
        """Reads data from HDF file
        Returns:
            array of `_act_dtype`
        """
        with h5py.File(filename, "r") as hfile:
            group = hfile[cls._hdf_path]
            record = np.empty(len(group), cls._act_dtype)
            for i, act in enumerate(group.values()):
                record[i]["sessions"] = np.array(
                    [cls._strip(item) for item in act], cls._session_dtype
                )
                for name, value in act.attrs.items():
                    record[i][name] = value
        return record

    @staticmethod
    def _strip(session) -> tuple:
        """Strips nans (from right side of all channels) added during hdf5 packaging
        Returns:
            tuple ready to be converted to `_session_dtype`
        """
        eeg, *rest = session
        ind = -next(i for i, value in enumerate(eeg[0, ::-1]) if not np.isnan(value))
        if ind == 0:
            ind = None
        return tuple((eeg[:, :ind], *rest))

    def get_data(self):
        for filename in os.listdir(self.data_dir):
            if ".hdf5" not in filename:
                continue
            row = self.markup[
                self.markup["filename"].str.contains(str(filename).strip())
            ]

            record = self.read_hdf(f"{self.data_dir}/{filename}")
            for i, act in enumerate(record):
                target = act["target"]
                is_train, sessions = act["is_train"], act["sessions"]

                if not is_train and i < 19:
                    target = row[str(i - 5)].values[0]

                epochs = self.transform_to_epochs(sessions)

                # eliminating sessions order
                stimuli = np.concatenate(sessions["stimuli"])
                self.eeg.append(
                    torch.Tensor(epochs).view(
                        epochs.shape[0], epochs.shape[2], epochs.shape[1]
                    )
                )
                self.labels.append(
                    torch.LongTensor([1 if stim == target else 0 for stim in stimuli])
                )
        return torch.cat(self.eeg, dim=0), torch.cat(self.labels, dim=0)

    def transform_to_epochs(self, sessions):
        eegs = self.transformation.fit_transform(sessions["eeg"])
        starts = sessions["starts"] // self.rate
        epochs = np.concatenate(
            [
                slice_epochs(eeg, sts, int(self.rate * 0.1), int(self.rate * 0.8))
                for eeg, sts in zip(eegs, starts)
            ]
        )
        return epochs

    def get_data_for_experiments(self, verbose=False):
        all_x, all_y = self.get_data()
        t = torch.arange(0, all_x.size(1), dtype=torch.float32)
        all_t = torch.stack([t] * all_x.size(0))

        total_seqs = all_x.shape[0]
        permutation = np.random.RandomState(42).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)

        self.test_x = all_x[permutation[:test_size]]
        self.test_y = all_y[permutation[:test_size]]
        self.test_t = all_t[permutation[:test_size]]

        self.train_x = all_x[permutation[test_size:]]
        self.train_y = all_y[permutation[test_size:]]
        self.train_t = all_t[permutation[test_size:]]

        self.feature_size = int(self.train_x.shape[-1])
        if verbose:
            print("train_x.shape: ", str(self.train_x.shape))
            print("train_y.shape: ", str(self.train_y.shape))
            print("Total number of train sequences: {}".format(self.train_x.shape[0]))
            print("Total number of test  sequences: {}".format(self.test_x.shape[0]))
