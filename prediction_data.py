import numpy as np
import matplotlib.pyplot as plt
from time_series import TimeSeries
from typing import Tuple, List, Union


class PredictionData:
    """Prediction data.

    Attributes:
        target (np.ndarray): The target values.
        features (dict): Each key is an abbreviation
            of a feature and each value is the
            corresponding feature values.

    """

    def __init__(self, filename: str) -> None:
        """Initiate an instance of PredictionData.

        Args:
            filename: The path of the input file.
        """
        self.features = {}
        with open(filename) as f:
            header = f.readline().strip().split(",")
            data = np.loadtxt(f, delimiter=",")
            column_num = len(data[0])
            self.target = data[:, column_num - 1]
            for column_index in range(0, column_num - 1):
                self.features[header[column_index]] = data[:, column_index]

    def scatter_plot(self, feature_abbrev: str) -> None:
        """Do scatter plot of a given feature and the target.

        Args:
            feature_abbrev: The abbreviation of a feature.

        """
        plt.figure(figsize=(10, 5))
        plt.scatter(self.features[feature_abbrev], self.target)
        feature_abbrev_label_relation = {
            "MedInc": "Median Income in Block Group",
            "HouseAge": "Median House Age in Block Group",
            "AveRooms": "Average Number of Rooms Per Household",
            "AveBedrms": "Average Number of Bedrooms Per Household",
            "Population": "Block Group Population",
            "AveOccup": "Average Number of Household Members",
            "Latitude": "Block Group Latitude",
            "Longitude": "Block Group Longitude"
        }
        x_label = feature_abbrev_label_relation[feature_abbrev]
        plt.xlabel(x_label)
        plt.ylabel("House Price ($100,000)")
        plt.title("Relationship Between California House Price And {}".format(x_label))
        plt.show()

    def show_means_and_stds(self) -> None:
        """Print the means and standard deviations."""
        for feature_abbrev in self.features:
            print(feature_abbrev)
            print("Mean: {}".format(self.features[feature_abbrev].mean()))
            print("Standard Deviation: {}".format(self.features[feature_abbrev].std()))
            print("========================================")
        print("Target\nMean: {}\nStandard Deviation: {}".format(self.target.mean(),
                                                                self.target.std()))

    @staticmethod
    def standardize_helper(nums: np.array) -> np.array:
        """Perform standardization for a single numpy array.

        Args:
            nums: The numpy array to be standardized.

        Returns:
            The standardized numpy array.

        """
        return (nums - nums.mean()) / nums.std()

    def standardize(self) -> None:
        """Standardize the features and target."""
        self.target = PredictionData.standardize_helper(self.target)
        for feature in self.features:
            self.features[feature] = PredictionData.standardize_helper(self.features[feature])

    def find_best_linear_regression_model(self) \
            -> Tuple[List[str],
                     List[List[Union[np.float, int]]],
                     np.ndarray,
                     dict]:
        """Find the best linear regression model.

        The best models are the models with
        lowest R-square.

        Returns:
            A tuple containing information of the
            best models and other models. The first
            element is the feature abbreviations of
            the best models. The second element is
            the betas of the best models. The third
            element is the lowest R-square. The four
            element is a dictionary contains the info
            of other non-best models, where each key is
            a feature abbreviation and each value
            is a tuple containing the betas and R-square.

        """
        lowest_r_square = float('inf')
        best_betas = []
        best_features = []
        models = {}
        for feature in self.features:
            x = self.features[feature]
            y = self.target
            current_betas, current_r_square = TimeSeries.linear_regression(x, y)
            models[feature] = (current_betas, current_r_square)
            if current_r_square <= lowest_r_square:
                lowest_r_square = current_r_square
                best_betas.append(current_betas)
                best_features.append(feature)
        return best_features, best_betas, lowest_r_square, models

    @staticmethod
    def show_model_info(models: dict):
        """Print information of a linear regression model

        Args:
            models: A dictionary contains the info
            of linear regression models, where each key is
            a feature abbreviation and each value
            is a tuple containing the betas and R-square.

        """
        for feature_abbrev in models:
            print(feature_abbrev)
            print("slope: {}".format(models[feature_abbrev][0][0]))
            print("y-intercept: {}".format(models[feature_abbrev][0][1]))
            print("R-square: {}".format(models[feature_abbrev][1][0]))
            print("========================================")

    def linear_regression_excluding_max(self, feature_abbrevs: List[str]) -> dict:
        """Perform linear regression for features excluding their maximum.

        This function does the following steps:
        For each feature:
        1. Delete the maximum value of the feature.
        2. Delete the corresponding target value of the maximum feature.
        3. Standardize the feature and the target.
        4. Perform linear regression.

        Args:
            feature_abbrevs: A list of feature abbreviations.

        Returns:
            A dictionary contains the info
            of linear regression models, where each key is
            a feature abbreviation and each value
            is a tuple containing the betas and R-square.

        """
        models = {}
        for feature_abbrev in feature_abbrevs:
            max_index = self.features[feature_abbrev].argmax()
            raw_x = np.delete(self.features[feature_abbrev], max_index)
            raw_y = np.delete(self.target, max_index)
            x = PredictionData.standardize_helper(raw_x)
            y = PredictionData.standardize_helper(raw_y)
            current_betas, current_r_square = TimeSeries.linear_regression(x, y)
            models[feature_abbrev] = (current_betas, current_r_square)
        return models
