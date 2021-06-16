from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def erpcov_ts_lr():
    """Obtains Riemannian features and classifies them with logregression"""
    return make_pipeline(
        ERPCovariances(estimator="oas"),
        TangentSpace(),
        LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced", penalty="l1"
        ),
    )
