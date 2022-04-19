from enum import Enum
from typing import Callable, Tuple, Union

import pandas as pd
from sklearn.base import is_classifier, is_regressor

import mlrun

from .._common import ModelType

# Type for a metric entry, can be passed as the metric function itself, as a callable object, a string of the name of
# the function and the full module path to the function to import. Arguments to use when calling the metric can be
# joined by wrapping it as a tuple:
MetricEntry = Union[Tuple[Union[Callable, str], dict], Callable, str]


class AlgorithmFunctionality(Enum):
    """
    An enum for the type of a machine learning algorithm. The algorithm types are based on the model's type and the
    predictions it will need to do. The types are chosen by the following table:
    __________________________________________________________________________________
    |                | Algorithm Functionality Name           | Outputs | Classes    |
    |________________|________________________________________|_________|____________|
    | Classification | Binary Classification                  | 1       | 2          |
    |                | Multiclass Classification              | 1       | >2         |
    |                | Multi Output Classification            | >1      | 2          |
    |                | Multi Output Multiclass Classification | >1      | >2         |
    |________________|________________________________________|_________|____________|
    | Regression     | Regression                             | 1       | Continuous |
    |                | Multi Output Regression                | >1      | Continuous |
    |________________________________________________________________________________|
    """

    # Classification:
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    MULTI_OUTPUT_CLASSIFICATION = "Multi Output Classification"
    MULTI_OUTPUT_MULTICLASS_CLASSIFICATION = "Multi Output Multiclass Classification"
    UNKNOWN_CLASSIFICATION = "Unknown Classification"

    # Regression:
    REGRESSION = "Regression"
    MULTI_OUTPUT_REGRESSION = "Multi Output Regression"
    UNKNOWN_REGRESSION = "Unknown Regression"

    @classmethod
    def get_algorithm_functionality(
        cls, model: ModelType, y: DatasetType = None
    ) -> "AlgorithmFunctionality":
        """
        Get the algorithm functionality according to the provided model and ground truth labels.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The algorithm functionality enum value.

        :raise MLRunInvalidArgumentError: If the model was not recognized to be a classifier or regressor.
        """
        # Convert the provided ground truths to DataFrame:
        if y is not None:
            y = to_dataframe(dataset=y)

        # Check for classification:
        if is_classifier(model):
            # Check if y is provided:
            if y is None:
                return cls.UNKNOWN_CLASSIFICATION
            # Check amount of columns:
            if len(y.columns) == 1:
                # Check amount of classes:
                if len(pd.unique(y.to_numpy().flatten())) <= 2:
                    return cls.BINARY_CLASSIFICATION
                return cls.MULTICLASS_CLASSIFICATION
            # More than one column, check amount of classes (2 columns means binary - 1 column for each class):
            if len(y.columns) == 2:
                return cls.MULTI_OUTPUT_CLASSIFICATION
            return cls.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION

        # Check for regression:
        if is_regressor(model):
            # Check if y is provided:
            if y is None:
                return cls.UNKNOWN_REGRESSION
            # Check amount of columns:
            if len(y.columns) == 1:
                return cls.REGRESSION
            return cls.MULTI_OUTPUT_REGRESSION

        # Unrecognized model:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Could not figure out if the given model '{type(model)}' is a classifier or regressor. Please contact us "
            f"on GitHub at https://github.com/mlrun/mlrun with the type of model that failed being recognized. You can "
            f"also use an explicit list of desired artifacts instead of calling the default method."
        )

    def is_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a classification model.

        :return: True if classification and False if not.
        """
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.UNKNOWN_CLASSIFICATION
        )

    def is_regression(self) -> bool:
        """
        Check whether this algorithm functionality is of a regression model.

        :return: True if regression and False if not.
        """
        return (
            self == AlgorithmFunctionality.REGRESSION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION
            or self == AlgorithmFunctionality.UNKNOWN_REGRESSION
        )

    def is_binary_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a binary classification model.

        :return: True if binary classification and False if not.
        """
        if self.is_regression():
            return False
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
        )

    def is_multiclass_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a multiclass classification model.

        :return: True if multiclass classification and False if not.
        """
        if self.is_regression():
            return False
        return (
            self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
        )

    def is_single_output(self) -> bool:
        """
        Check whether this algorithm functionality is predicting a single output.

        :return: True if predicting a single output and False if not.
        """
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.REGRESSION
        )

    def is_multi_output(self) -> bool:
        """
        Check whether this algorithm functionality is predicting multiple outputs.

        :return: True if predicting multiple outputs and False if not.
        """
        return (
            self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION
        )
