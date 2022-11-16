from typing import Dict, List, Union

import numpy as np
import pandas as pd

import mlrun
from mlrun.feature_store import FeatureSet, FeatureVector
from mlrun.model import ModelObj
from mlrun.artifacts import Artifact

from .monitors import MonitorType

ModelType = object
MLRunObjectType = Union[Artifact, FeatureSet, FeatureVector]

EXAMPLE = {
    "key": "monitor-policy_example",
    "is_realtime_serving": True,
    "global_kwargs": {
        "x": "12gfd3-453453425g34-5443gff",
        "y_true": "234d432f3-534453345544-5g2f",
    },
    "monitors": [
        {
            "key": "data-drift-1",
            "monitor_class": "mlrun.monitoring.monitors.DataDistributionDriftMonitor",
            "run_time_function": "fsfsdf-534-fsdfdss-4534",
            "schedule": "2h",
            "local_kwargs": {
                "bins": 10,
                "distribution_metrics": [
                    ("total_variance_distance", {}),
                    ("kullback_leibler_divergence", {"inf_cap": 10.0}),
                ],
            },
        },
        {
            "key": "data-drift-2",
            "monitor_class": "mlrun.monitoring.monitors.DataDistributionDriftMonitor",
            "schedule": "30m",
            "local_kwargs": {
                "bins": 60,
                "statistics_metrics": [("min", {}), ("max", {})],
            },
        },
    ],
}


class MonitorPolicy(ModelObj):
    _dict_fields = [
        "key",
        "monitors",
        "model",
        "x_ref",
        "y_ref",
        "x",
        "y_pred",
        "y_true",
        "schedule",
    ]

    def __init__(
        self,
        key: str = None,
        monitor_type: Union[MonitorType, str] = None,
        is_realtime_serving: bool = None,
        model: str = None,
        x_ref: Union[pd.DataFrame, str] = None,
        y_ref: Union[pd.DataFrame, str] = None,
        y_columns_names: List[str] = None,
        y_pred_suffix: str = None,
        y_true_suffix: str = None,
        # Online mode:
        endpoint_id: str = None,
        tsdb_kwargs: Dict[str, str] = None,
        from_date: str = None,
        to_date: str = None,
        # Offline mode:
        x: Union[pd.DataFrame, str] = None,
        y_pred: Union[pd.DataFrame, str] = None,
        y_true: Union[pd.DataFrame, str] = None,
        monitor_id: str = None,
    ):
        pass
