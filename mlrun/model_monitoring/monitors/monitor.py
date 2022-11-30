from typing import Callable, Dict, List, Union, Tuple, Optional, NamedTuple
from types import FunctionType
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
from datetime import datetime
import re
import numpy as np
import pandas as pd

import mlrun
import mlrun.feature_store as mlrun_fs
from mlrun.model import ModelObj
from mlrun.artifacts import Artifact, ModelArtifact
from mlrun.datastore import is_store_uri, store_manager


ModelType = object
MLRunObjectType = Union[
    Artifact, ModelArtifact, mlrun_fs.FeatureSet, mlrun_fs.FeatureVector
]


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
#       * y are the model outputs, aka y_pred (aka predictions)
#       * Possible overriding: __init__, validate, load ,run
class Monitor(ABC):

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        endpoint_id: str,
        monitor_type: Union[MonitorType, str],
        is_realtime_serving: bool,
        model: str = None,
        x_ref: Union[pd.DataFrame, str] = None,
        y_ref: Union[pd.DataFrame, str] = None,
        y_columns_names: List[str] = None,
        index: str = "index",
        selected_columns: List[str] = None,
        # Online mode:
        tsdb_kwargs: Dict[str, str] = None,
        from_date: str = None,
        to_date: str = None,
        # Offline mode:
        x: Union[pd.DataFrame, str] = None,
        y: Union[pd.DataFrame, str] = None,
        y_true: Union[pd.DataFrame, str] = None,
        monitor_id: str = None,
    ):
        # Store given common properties:
        self._context = context
        self._endpoint_id = endpoint_id
        self._monitor_type = MonitorType(monitor_type)
        self._is_realtime_serving = is_realtime_serving
        self._model = model
        self._x_ref = x_ref
        self._y_ref = y_ref
        self._y_columns_names = y_columns_names
        self._index = index
        self._selected_columns = selected_columns

        # Store online mode properties:
        self._tsdb_kwargs = tsdb_kwargs
        self._from_date = from_date
        self._to_date = to_date

        # Store offline mode properties:
        self._x = x
        self._y = y
        self._y_true = y_true
        self._monitor_id = monitor_id

        # Common extra data:
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

        # Initialize the mlrun objects and inputs from context:
        self._init_from_mlrun()

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
    def y(self) -> pd.DataFrame:
        return self._y

    @property
    def y_true(self) -> pd.DataFrame:
        return self._y_true

    @property
    def feature_importance(self) -> Dict[str, float]:
        return self._feature_importance

    @property
    def y_columns_names(self) -> List[str]:
        return self._y_columns_names

    @property
    def index(self) -> str:
        return self._index

    @property
    def selected_columns(self) -> List[str]:
        return self._selected_columns

    @property
    def context(self):
        return self._context

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
        raise RuntimeError(
            "The generic loading model `load_model` method is not implemented. Please consider inheriting this class "
            "and override the method or simply write its logic in the `load` method instead.\n"
            "\n"
            "You may use the `mlrun.frameworks.{your-framework-of-choice}`'s ModelHandler class or call the "
            "`mlrun.artifacts.get_model` function."
        )

    def load_x_ref(self):
        # Make sure `x_ref` is available for loading:
        if self._x_ref is None:
            raise mlrun.errors.MLRunRuntimeError(
                "Cannot load `x_ref`."
                ""
                "`x_ref` can be provided manually to the class initializer or via the extra data of a model artifact "
                "(looking for key: 'ref_data' in the extra data dictionary)."
            )

        # Load the dataset:
        self._x_ref = self._load_dataset(dataset=self._x_ref)

        # Look for y columns to split it to `y_ref` if it was not provided. If it was, ignore all y columns:
        if self.y_columns_names is not None:
            # Look for the y columns:
            y_columns = set(self.y_columns_names) & set(self._x_ref.columns)
            if len(y_columns) != 0:
                # Update `y_ref` in case it was not provided:
                if self.y_ref is None:
                    self._y_ref = self._x_ref[y_columns]
                    self.load_y_ref()
                # Remove the y columns from the x dataset:
                self._x_ref.drop(columns=y_columns, inplace=True)

        # Clear the non-selected columns by the user:
        if self.selected_columns is not None:
            # Leave only selected columns:
            self._x_ref = self._x_ref[
                set(self.selected_columns) & set(self._x_ref.columns)
            ]

    def load_y_ref(self):
        # Make sure `y_ref` is available for loading:
        if self._y_ref is None:
            # `y_ref` was not provided, try to get it from the `x_ref` dataset:
            self._context.logger.info(
                "`y_ref` was not provided, looking for `y_ref` in `x_ref`"
            )
            self.load_x_ref()
            if self._y_ref is None:
                raise mlrun.errors.MLRunRuntimeError(
                    f"Cannot load `y_ref`.\n"
                    f"\n"
                    f"`y_ref` can be provided in the following ways:\n"
                    f"* Via the `y_ref` attribute to the class initializer.\n"
                    f"* Via the `x_ref` attribute to the class initializer. The columns for `y_ref` will be taken "
                    f"from the `y_columns_names` attribute. `y_columns_names` can be also read from the model artifact "
                    f"if it was logged with it. The available `y_columns_names` now are: {self.y_columns_names}"
                    f"* Via the extra data of a model artifact (looking for key: 'ref_data' in the extra data "
                    f"dictionary). The `y_columns_names` must match to the columns in the 'ref_data' of the model."
                )
            return

        # Load the dataset:
        self._y_ref = self._load_dataset(dataset=self._y_ref)

        # Clear the non-selected columns by the user:
        if self.selected_columns is not None:
            # Leave only selected columns:
            self._y_ref = self._y_ref[
                set(self.selected_columns) & set(self._y_ref.columns)
            ]

    def load_x(self):
        # Make sure `y_ref` is available for loading:
        if self.is_realtime_serving:
            return
        if self._x is None:
            raise mlrun.errors.MLRunRuntimeError(
                "Cannot load `x`. `x` was not provided."
            )

        # Load the dataset:
        self._x = self._load_dataset(dataset=self._x)

        # Look for y columns to split it to `y_ref` if it was not provided. If it was, ignore all y columns:
        if self.y_columns_names is not None:
            # Look for the y columns:
            y_columns = set(self.y_columns_names) & set(self._x.columns)
            if len(y_columns) != 0:
                # Update `y_pred` in case it was not provided:
                if self.y is None:
                    self._y = self._x[y_columns]
                    self.load_y()
                # Remove the y pred columns from the x dataset:
                self._x.drop(columns=y_columns, inplace=True)

        # Clear the non-selected columns by the user:
        if self.selected_columns is not None:
            # Leave only selected columns:
            self._x = self._x[set(self.selected_columns) & set(self._x.columns)]

    def load_y(self):
        # Make sure `y_ref` is available for loading:
        if self._y is None:
            # `y_pred` was not provided, try to get it from the `x` dataset:
            self._context.logger.info(
                "`y_pred` was not provided, looking for `y_pred` in `x`"
            )
            self.load_x()
            if self._y is None:
                raise mlrun.errors.MLRunRuntimeError(
                    f"Cannot load `y_pred`.\n"
                    f"\n"
                    f"`y_pred` can be provided in the following ways:\n"
                    f"* Via the `y_pred` attribute to the class initializer.\n"
                    f"* Via the `x` attribute to the class initializer. The columns for `y_pred` will be taken "
                    f"from the `y_columns_names` attribute. `y_columns_names` can be also read from the model artifact "
                    f"if it was logged with it. The available `y_columns_names` now are: {self.y_columns_names}"
                )
            return

        # Load the dataset:
        self._y = self._load_dataset(dataset=self._y)

        # Clear the non-selected columns by the user:
        if self.selected_columns is not None:
            # Leave only selected columns:
            self._y = self._y[
                set(self.selected_columns) & set(self._y_ref.columns)
            ]

    def load_y_true(self):
        # Make sure `y_ref` is available for loading:
        if self._y_true is None:
            raise mlrun.errors.MLRunRuntimeError(
                f"Cannot load `y_true`. `y_true was not provided."
            )

        # Load the dataset:
        self._y_true = self._load_dataset(dataset=self._y_true)

        # Clear the non-selected columns by the user:
        if self.selected_columns is not None:
            # Leave only selected columns:
            self._y_true = self._y_true[
                set(self.selected_columns) & set(self._y_ref.columns)
            ]

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
        """
        1. Validate all offline arguments.
        """
        pass

    def _validate_online_mode(self):
        """
        1. Validate all online arguments.
        """
        pass

    def _init_offline_mode(self):
        """
        1. Create a monitor ID if not provided.
        """
        if self._monitor_id is None:
            self._monitor_id = hashlib.sha224(str(datetime.now()).encode()).hexdigest()

    def _init_online_mode(self):
        """
        1. Init the TSDB API object to connect to the configured TSDB (V3IO's TSDB or Prometheus).
        """
        pass

    def _init_from_mlrun(self):
        # Store all context parameters (for extra parameters the user may pass):
        for parameter_name, parameter_value in {
            **self._context.inputs,
            **self._context.parameters,
        }.items():
            if hasattr(self, parameter_name):
                continue
            setattr(self, parameter_name, parameter_value)

        # Get the MLRun object of all the stored properties (if they are a store path):
        for attribute_name, attribute in self.__dict__.items():
            if isinstance(attribute, str) and is_store_uri(attribute):
                self.load_mlrun_object(url=attribute, key=attribute_name.lstrip("_"))

        # Check if a model url was given:
        if "model" not in self._mlrun_objects:
            return

        # Get the model artifact:
        model_artifact: ModelArtifact = self.get_mlrun_object(key="model")

        # Look for reference data in the extra data in case 'x_ref' was not provided:
        if self.x_ref is None:
            self._x_ref = model_artifact.spec.extra_data.get("ref_data", None)

        # Look for feature importance in the extra data:
        self._feature_importance = model_artifact.spec.extra_data.get(
            "feature_importance", None
        )

        # Get the y column names (outputs) from the model artifact if available:
        if self.y_columns_names is None and len(model_artifact.spec.outputs) != 0:
            self._y_columns_names = [
                output.name for output in model_artifact.spec.outputs
            ]

    @staticmethod
    def _load_dataset(
        dataset: Union[str, mlrun.DataItem, mlrun_fs.FeatureVector, pd.DataFrame]
    ) -> pd.DataFrame:
        if isinstance(dataset, mlrun_fs.FeatureVector):
            dataset = mlrun_fs.get_offline_features(
                feature_vector=dataset, with_indexes=True
            )

        # If it's a loaded dataframe, simply return:
        if isinstance(dataset, pd.DataFrame):
            return dataset

        # If it's a path to a file or a uri of a dataset artifact, get the DataItem:
        if isinstance(dataset, str):
            dataset = mlrun.get_dataitem(url=dataset)

        # Parse into pandas dataframe and return:
        return dataset.as_df()

    @classmethod
    def handler(
        cls,
        context: mlrun.MLClientCtx,
        policy: Union[dict, str],
        **kwargs
    ):
        # Initialize the monitor class from policy:
        monitor = cls.from_policy(context=context, policy=policy)

        # Load the user's required assets before the run:
        monitor.load()

        # Run the monitoring process:
        results = monitor.run()

        # Log the results
        context.log_results(results=results)
