from typing import Callable, Dict, List, Union, Tuple
from types import FunctionType
from abc import ABC

import numpy as np
import pandas as pd

import mlrun
from .monitor import Monitor, MonitorType


class StatisticMetrics:
    # General:
    @staticmethod
    def count(data: pd.DataFrame) -> pd.DataFrame:
        return data.count()

    @staticmethod
    def min(data: pd.DataFrame) -> pd.DataFrame:
        return data.min()

    @staticmethod
    def max(data: pd.DataFrame) -> pd.DataFrame:
        return data.max()

    # Location:
    @staticmethod
    def mean(data: pd.DataFrame) -> pd.DataFrame:
        return data.mean()

    @staticmethod
    def median(data: pd.DataFrame) -> pd.DataFrame:
        return data.median()

    @staticmethod
    def mode(data: pd.DataFrame) -> pd.DataFrame:
        return data.mode()

    # Spread:
    @staticmethod
    def standard_deviation(data: pd.DataFrame) -> pd.DataFrame:
        return data.std()

    @staticmethod
    def variance(data: pd.DataFrame) -> pd.DataFrame:
        return data.var()

    @staticmethod
    def mean_absolute_error(data: pd.DataFrame) -> pd.DataFrame:
        return data.mad()

    # Shape:
    @staticmethod
    def skewness(data: pd.DataFrame) -> pd.DataFrame:
        return data.skew()

    @staticmethod
    def kurtosis(data: pd.DataFrame) -> pd.DataFrame:
        return data.kurt()


class DistributionDistanceMetrics:
    @staticmethod
    def total_variance_distance(p: np.ndarray, q: np.ndarray) -> float:
        return np.sum(np.abs(p - q)) / 2

    @staticmethod
    def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        return np.sqrt(1 - np.sum(np.sqrt(p * q)))

    @staticmethod
    def kullback_leibler_divergence(
        p: np.ndarray, q: np.ndarray, inf_cap: float = 10.0, kld_scaling: float = 1e-4
    ) -> float:
        p_q = np.sum(
            np.where(
                p != 0,
                p * np.log(p / np.where(q != 0, q, kld_scaling)),
                0,
            )
        )
        q_p = np.sum(
            np.where(
                q != 0,
                q * np.log(q / np.where(p != 0, p, kld_scaling)),
                0,
            )
        )
        result = p_q + q_p
        if inf_cap:
            return inf_cap if result == float("inf") else result
        return result


class DistributionDriftMonitor(Monitor, ABC):

    FEATURE_IMPORTANCE_WEIGHTS_VALUE = "feature_importance"

    _DEFAULT_STATISTIC_METRICS = [
        ("count", {}),
        ("mean", {}),
        ("std", {}),
        ("min", {}),
        ("max", {}),
    ]
    _DEFAULT_DISTRIBUTION_DISTANCE_METRICS = [
        ("total_variance_distance", {}),
        ("hellinger_distance", {}),
        ("kullback_leibler_divergence", {}),
    ]

    def __init__(
        self,
        statistic_metrics: List[
            Tuple[str, dict],
        ] = None,  # The functions are of type: Callable[[pd.DataFrame], pd.DataFrame]
        distribution_distance_metrics: List[
            Tuple[str, dict]
        ] = None,  # The functions are of type: Callable[[pd.Series, pd.Series], float]

        histogram_bins: int = 20,
        weights: Union[pd.DataFrame, Dict[str, float], str] = None,
    ):
        self._statistic_metrics = self._parse_metrics(
            metrics_class=StatisticMetrics,
            metrics_list=(
                statistic_metrics
                if statistic_metrics is not None
                else self._DEFAULT_STATISTIC_METRICS
            ),
        )
        self._distribution_distance_metrics = self._parse_metrics(
            metrics_class=DistributionDistanceMetrics,
            metrics_list=(
                distribution_distance_metrics
                if distribution_distance_metrics is not None
                else self._DEFAULT_DISTRIBUTION_DISTANCE_METRICS
            ),
        )
        self._histogram_bins = histogram_bins
        self._weights = weights

        self._x_ref_statistics = None
        self._y_ref_statistics = None

    @property
    def x_ref_statistics(self) -> dict:
        return self._x_ref_statistics

    @property
    def y_ref_statistics(self) -> dict:
        return self._y_ref_statistics

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights

    def load_weights(self):
        if self._weights is None:
            return

        if isinstance(self._weights, str):
            if self._weights == self.FEATURE_IMPORTANCE_WEIGHTS_VALUE:
                if self.get_mlrun_object(key="model") is None:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "Weights cannot be set to 'feature_importance' as there is no model provided to the monitor."
                    )
                self.load_feature_importance()
                if self.feature_importance is None:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "Weights cannot be set to 'feature_importance' as there are no feature importance available in "
                        "the model object."
                    )
                self._weights = self.feature_importance
            else:
                self._weights = self.load_dataset(url=self._weights)

            if isinstance(self._weights, pd.DataFrame):
                self._weights = self._weights.to_dict()

    def calculate_dataset_histograms(
        self, dataset: pd.DataFrame
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        return {
            column: np.histogram(dataset[column], bins=self._histogram_bins)
            for column in dataset.columns
        }

    def calculate_statistics(
        self, dataset: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        # Set up lists to hold the statistics calculations data frames and the metric names - columns:
        calculations: List[pd.DataFrame] = []
        columns: List[str] = []

        # Go over the statistics metrics and calculate while collecting the columns names:
        for metric_name, (metric_function, metric_kwargs) in self._statistic_metrics:
            calculations.append(metric_function(dataset, **metric_kwargs))
            columns.append(metric_name)

        # Concatenate the results into a single data frame:
        result = pd.concat(objs=calculations, axis=1)
        result.columns = columns

        # Return the dictionary representation of the results:
        return result.T.to_dict()

    def calculate_distribution_distances(
        self,
        dataset_histograms_1: Dict[str, Tuple[np.ndarray, int]],
        dataset_histograms_2: Dict[str, Tuple[np.ndarray, int]],
    ) -> Dict[str, Dict[str, float]]:
        # Set up a list to hold the metrics calculations data frames:
        calculations = []

        # Go over the metric and call them for each feature:
        for metric_name, (
            metric_function,
            metric_kwargs,
        ) in self._distribution_distance_metrics:
            # Set up a list to hold the metric scores across all features:
            metric_scores: List[float] = []
            # Go over the columns (features) and calculate the score using the metric function:
            for column in dataset_histograms_1:
                metric_scores.append(
                    metric_function(
                        dataset_histograms_1[column][0],
                        dataset_histograms_2[column][0],
                        **metric_kwargs
                    )
                )
            # Collect the scores of the metric as a data frame:
            calculations.append(
                pd.DataFrame(
                    metric_scores,
                    index=list(dataset_histograms_1.keys()),
                    columns=[metric_name],
                )
            )

        # Concatenate the results into a single data frame:
        calculations = pd.concat(objs=calculations, axis=1)

        # Return the dictionary representation of the results:
        return calculations.T.to_dict()

    def calculate_drift_score(self):
        pass

    def load(self):
        super().load()
        if self._weights is not None:
            self.load_weights()

    @staticmethod
    def _parse_metrics(
        metrics_class: type, metrics_list: List[Tuple[str, dict]]
    ) -> Dict[str, Tuple[FunctionType, dict]]:
        return {
            metric_name: (getattr(metrics_class, metric_name), metric_kwargs)
            for metric_name, metric_kwargs in metrics_list
        }


class DataDistributionDriftMonitor(DistributionDriftMonitor):
    """
    x with x_ref
    """

    def load(self):
        super().load()
        self.load_x_ref()
        self.load_x()

    def run(self):
        x_statistics = self.calculate_statistics(dataset=self.x)
        x_ref_statistics = self.calculate_statistics(dataset=self.x_ref)

        x_histogram = self.calculate_dataset_histograms(dataset=self.x)
        x_ref_histogram = self.calculate_dataset_histograms(dataset=self.x_ref)

        distribution_distances = self.calculate_distribution_distances(
            dataset_histograms_1=x_histogram, dataset_histograms_2=x_ref_histogram
        )
        if self.weights:
            distribution_distances = self.apply_weights(results=distribution_distances)


class ConceptDistributionDriftMonitor(DistributionDriftMonitor):
    """
    y_pred with y_ref
    y_true with y_ref
    """

    def load(self):
        super().load()
        self.load_y_ref()
        if self.y_pred is not None:
            self.load_y_pred()
        if self.y_true is not None:
            self.load_y_true()

    def run(self):
        pass
