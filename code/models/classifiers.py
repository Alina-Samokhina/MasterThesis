import numpy as np
from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from ..transforms import Flattener


# fmt: off
# from https://eeg-notebooks.readthedocs.io/en/latest/visual_p300.html
# and https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
clfs_full = {
    'LR': (
        make_pipeline(Flattener(), LogisticRegression(solver='liblinear')),
        {
            'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
        },
    ),
    'SVM': (
        make_pipeline(Flattener(), SVC(C=55, kernel='rbf', probability=True)),
        {
            'svc__C': np.exp(np.linspace(-4, 4, 5)),
            'svc__kernel': ('linear', 'rbf'),
        },
    ),
    'ERPCov TS LR': (
        make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(), LogisticRegression(solver='liblinear')),
        {
            'erpcovariances__estimator': ('lwf', 'oas'),
            'logisticregression__C': np.exp(np.linspace(-4, 4, 5)),
        },
    ),}


for name, (clf, _) in clfs_full.items():
    clf.name = name


def get_classifier():
    lr_params = {
        'solver': 'liblinear',
        'max_iter': 1000,
        'C': 1.0,
        'class_weight': 'balanced',
        'penalty': 'l1',
    }
    return make_pipeline(
        ERPCovariances(estimator='oas'),
        TangentSpace(),
        LogisticRegression(**lr_params),
    )
# fmt: on
