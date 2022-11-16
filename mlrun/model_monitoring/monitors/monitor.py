from typing import Callable, Dict, List, Union, Tuple, Optional, NamedTuple
from types import FunctionType
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

import mlrun
from mlrun.feature_store import FeatureSet, FeatureVector
from mlrun.model import ModelObj
from mlrun.artifacts import Artifact
from mlrun.datastore import is_store_uri, store_manager


ModelType = object
MLRunObjectType = Union[Artifact, FeatureSet, FeatureVector]


def get_artifact(uri: str) -> Artifact:
    pass


def get_extra_data(
    artifact: Artifact,
) -> Dict[str, Union[str, float, int, bool, mlrun.DataItem]]:
    pass


def load_model(path_or_uri: str) -> ModelType:
    pass


def load_dataset(path_or_uri: str) -> pd.DataFrame:
    pass


# TODO: Temporary for the tsdb kwargs.
class TSDBKeys:
    PATH = "path"
    TABLE_NAME = "table_name"
    TOKEN = "token"
    ADDRESS = "address"
    CONTAINER = "container"
    BACKEND = "backend"


class MonitorType(Enum):
    DATA_INTEGRITY = "data_integrity"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_PERFORMANCE = "model_performance"
    MODEL_USAGE = "model_usage"
    CUSTOM = "custom"


class MonitorSpec(ModelObj):

    _dict_fields = ["monitor_type", "is_realtime_serving", "monitor_id"]

    def __init__(
        self,
        monitor_type: Union[MonitorType, str],
        is_realtime_serving: bool,
        monitor_id: str = None,
        **kwargs,
    ):
        pass


# TODO: Mention in doc string that:
#       * x_ref and y_ref are the reference data (aka baseline data)
#       * y_true is the ground truth (aka actual values)
class Monitor(ABC):
    def __init__(
        self,
        context: mlrun.MLClientCtx,
        monitor_type: Union[MonitorType, str],
        is_realtime_serving: bool,
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
        # Store given common properties:
        self._context = context
        self._monitor_type = MonitorType(monitor_type)
        self._is_realtime_serving = is_realtime_serving
        self._model = model
        self._x_ref = x_ref
        self._y_ref = y_ref
        self._y_columns_names = y_columns_names
        self._y_pred_suffix = y_pred_suffix
        self._y_true_suffix = y_true_suffix

        # Store online mode properties:
        self._endpoint_id = endpoint_id
        self._tsdb_kwargs = tsdb_kwargs
        self._from_date = from_date
        self._to_date = to_date

        # Store offline mode properties:
        self._x = x
        self._y_pred = y_pred
        self._y_true = y_true
        self._monitor_id = monitor_id

        # Common extra data:
        self._x_ref_statistics = None
        self._y_ref_statistics = None
        self._feature_importance = None

        # Set up handlers:
        self._mlrun_objects: Dict[str, MLRunObjectType] = {}

        # Validate common properties:
        self.validate()

        # Continue initialization by mode:
        if self._is_realtime_serving:  # Online mode
            self._init_online_mode()
        else:  # Offline mode
            self._init_offline_mode()

        # Get the MLRun objects:
        self._get_mlrun_objects()

    @property
    def is_realtime_serving(self) -> bool:
        return self._is_realtime_serving

    @property
    def model(self) -> ModelType:
        return self._model

    @property
    def x_ref(self) -> pd.DataFrame:
        return self._x_ref

    @property
    def y_ref(self) -> pd.DataFrame:
        return self._y_ref

    @property
    def x(self) -> pd.DataFrame:
        return self._x

    @property
    def y_pred(self) -> pd.DataFrame:
        return self._y_pred

    @property
    def y_true(self) -> pd.DataFrame:
        return self._y_true

    @property
    def feature_importance(self) -> Dict[str, float]:
        return self._feature_importance

    def get_mlrun_object(self, key: str) -> Union[MLRunObjectType, None]:
        return self._mlrun_objects.get(key, None)

    def get_secret(self, key: str) -> str:
        return self._context.get_secret(key=key)

    @classmethod
    def from_policy(cls, context: mlrun.MLClientCtx, policy: dict) -> "Monitor":
        return cls(context=context, **policy)

    def to_policy(self) -> dict:
        pass

    def load_model(self):
        raise NotImplementedError(
            "The generic loading model `load_model` method is yet to be implemented. Please consider inheriting this "
            "class and override the method or simply write its logic in the `load` method instead."
            ""
            "You may use the `mlrun.framework.{your-framework-of-choice}`'s ModelHandler class or call the "
            "`mlrun.artifacts.get_model` function."
        )

    def load_x_ref(self):
        pass

    def load_y_ref(self):
        # Check if it was already loaded as part of `x_ref`:
        if isinstance(self._y_ref, pd.DataFrame):
            return

    def load_x(self):
        if self.is_realtime_serving:
            return
        if isinstance(self._x, str):
            self._x = mlrun.get_dataitem(url=self._x).as_df()
        elif not isinstance(self._x, pd.DataFrame):
            raise mlrun.errors.MLRunInvalidArgumentError()

    def load_y_pred(self):
        # Check if it was already loaded as part of `x_ref`:
        if isinstance(self._y_pred, pd.DataFrame):
            return

    def load_y_true(self):
        # Check if it was already loaded as part of `x_ref`:
        if isinstance(self._y_true, pd.DataFrame):
            return

    def load_feature_importance(self):
        # Check if a model uri was given:
        if "model" not in self._mlrun_objects:
            raise mlrun.errors.MLRunRuntimeError(
                "Cannot load 'feature importance' dictionary if a model url was not provided."
            )

        # Get the feature importance out of the model artifact's extra data:
        model_extra_data = self.get_mlrun_object(key="model").spec.extra_data
        if "feature_importance" in model_extra_data:
            self._feature_importance = model_extra_data["feature_importance"]

    def load_mlrun_object(self, url: str, key: str):
        self._mlrun_objects[key] = store_manager.get_store_artifact(url=url)[0]

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def run(self) -> Optional[Dict[str, Union[float, int]]]:
        pass

    def validate(self):
        # Validate given properties:

        # Continue validation by mode:
        if self._is_realtime_serving:  # Online mode
            self._validate_online_mode()
        else:  # Offline mode
            self._validate_offline_mode()

    def _validate_offline_mode(self):
        pass

    def _validate_online_mode(self):
        pass

    def _init_offline_mode(self):
        """
        1. Prepare all given datasets to be DataItems.
        2. Create a monitor ID if not provided.
        """
        if self._monitor_id is None:
            self._monitor_id = hashlib.sha224(str(datetime.now()).encode()).hexdigest()

    def _init_online_mode(self):
        """ """
        pass

    def _get_mlrun_objects(self):
        for attribute_name, attribute in self.__dict__.items():
            if isinstance(attribute, str) and is_store_uri(attribute):
                self.load_mlrun_object(url=attribute, key=attribute_name.lstrip("_"))

    @classmethod
    def handler(
        cls,
        context: mlrun.MLClientCtx,
        policy: Union[dict, str],
    ):
        # Initialize the monitor class from policy:
        monitor = cls.from_policy(context=context, policy=policy)

        # Load the user's required assets before the run:
        monitor.load()

        # Run the monitoring process:
        results = monitor.run()

        # Log the results
        context.log_results(results=results)
