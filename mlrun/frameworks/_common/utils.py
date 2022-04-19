from pathlib import Path
from typing import List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import scipy.sparse.base

import mlrun
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem

# Generic types:
ModelType = TypeVar(
    "ModelType"
)  # A generic model type in a handler / interface (examples: tf.keras.Model, torch.Module).
IOSampleType = TypeVar(
    "IOSampleType"
)  # A generic inout / output samples for reading the inputs / outputs properties.
MLRunInterfaceableType = TypeVar(
    "MLRunInterfaceableType"
)  # A generic object type for what can be wrapped with a framework MLRun interface (examples: xgb, xgb.XGBModel).

# Common types:
PathType = Union[str, Path]  # For receiving a path from 'pathlib' or 'os.path'.
TrackableType = Union[str, bool, float, int]  # All trackable values types for a logger.
ExtraDataType = Union[
    str, bytes, Artifact, DataItem
]  # Types available in the extra data dictionary of an artifact.
DatasetType = Union[
    list, dict, np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.base.spmatrix
]  # A type for all the supported dataset types


def to_array(dataset: DatasetType) -> np.ndarray:
    """
    Convert the given dataset to np.ndarray.

    :param dataset: The dataset to convert. Must be one of {pd.DataFrame, pd.Series, scipy.sparse.base.spmatrix, list,
                    dict}.

    :return: The dataset as a ndarray.

    :raise MLRunInvalidArgumentError: If the dataset type is not supported.
    """
    if isinstance(dataset, np.ndarray):
        return dataset
    if isinstance(dataset, (pd.DataFrame, pd.Series)):
        return dataset.to_numpy()
    if isinstance(dataset, scipy.sparse.base.spmatrix):
        return dataset.toarray()
    if isinstance(dataset, list):
        return np.array(dataset)
    if isinstance(dataset, dict):
        return np.array(list(dataset.values()))
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"Could not convert the given dataset into a numpy ndarray. Supporting conversion from: "
        f"'pandas.DataFrame', 'pandas.Series', 'scipy.sparse.base.spmatrix', list and dict. The given dataset was of "
        f"type: '{type(dataset)}'"
    )


def to_dataframe(dataset: DatasetType) -> pd.DataFrame:
    """
    Convert the given dataset to pd.DataFrame.

    :param dataset: The dataset to convert. Must be one of {np.ndarray, pd.Series, scipy.sparse.base.spmatrix, list,
                    dict}.

    :return: The dataset as a DataFrame.

    :raise MLRunInvalidArgumentError: If the dataset type is not supported.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if isinstance(dataset, (np.ndarray, pd.Series, list, dict)):
        return pd.DataFrame(dataset)
    if isinstance(dataset, scipy.sparse.base.spmatrix):
        return pd.DataFrame.sparse.from_spmatrix(dataset)
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"Could not convert the given dataset into a pandas DataFrame. Supporting conversion from: "
        f"'numpy.ndarray', 'pandas.Series', 'scipy.sparse.base.spmatrix' list and dict. The given dataset was of type: "
        f"'{type(dataset)}'"
    )


def concatenate_x_y(
    x: DatasetType,
    y: DatasetType = None,
    y_columns: Union[List[str], List[int]] = None,
    default_y_column_prefix: str = "y_",
) -> Tuple[pd.DataFrame, Union[Union[List[str], List[int]], None]]:
    """
    Concatenating the provided x and y data into a single pd.DataFrame, casting from np.ndarray and renaming y's
    original columns if 'y_columns' was not provided. The concatenated dataset index level will be reset to 0
    (multi-level indexes will be dropped using pandas 'reset_index' method).

    :param x:                       A collection of inputs to a model.
    :param y:                       A collection of ground truth labels corresponding to the inputs.
    :param y_columns:               List of names or indices to give the columns of the ground truth labels.
    :param default_y_column_prefix: A default value to join the y columns in case one of them is found in x (so there
                                    won't be any duplicates). Defaulted to: "y_".
    :return: A tuple of:
             [0] = The concatenated x and y as a single DataFrame.
             [1] = The y columns names / indices.
    """
    # Cast x to a DataFrame (from np.ndarray and pd.Series):
    x = to_dataframe(dataset=x)
    if y is None:
        # Reset the indices levels:
        x = x.reset_index(drop=True)
        return x, None

    # Cast y to a DataFrame (from np.ndarray and pd.Series):
    y = to_dataframe(dataset=y)

    # Check if y's columns are given, if not set the default avoiding duplicates with x's columns:
    if y_columns is None:
        y_columns = [
            column if column not in x.columns else f"{default_y_column_prefix}{column}"
            for column in list(y.columns)
        ]

    # Override the columns with the names the user provided:
    y.columns = y_columns

    # Concatenate the x and y into a single dataset:
    dataset = pd.concat([x, y], axis=1)

    # Reset the indices levels:
    dataset.reset_index(drop=True, inplace=True)

    return dataset, y_columns
