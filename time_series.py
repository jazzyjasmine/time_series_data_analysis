import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List, Union


class TimeSeries:
    """Time Series data.

    Attributes:
        timestamps (np.chararray): The timestamps of the data.
            For example, "2014-01" is a timestamp.
        metrics (np.ndarray): The metrics of the data.
        moving_averages (dict): Each key is a window size of the
            moving average and each value is the moving average
            values under the given window size.

    """

    def __init__(self,
                 filename: str,
                 months_column_index: int,
                 years_column_index: int,
                 metrics_column_index: int) -> None:
        """Initiate an instance of TimeSeries class.

        Column index starts from 0.

        Args:
            filename: The path of the input file.
            months_column_index: The column index of months.
            years_column_index: The column index of years.
            metrics_column_index: The column index of metrics.
        """
        self.timestamps, self.metrics = self.load_data(filename, months_column_index, years_column_index,
                                                       metrics_column_index)
        self.moving_averages = {}

    @staticmethod
    def load_data(filename: str,
                  months_column_index: int,
                  years_column_index: int,
                  metrics_column_index: int) -> Tuple[np.chararray, np.ndarray]:
        """Load data from an input file containing time series data.

        Column index starts from 0.

        Args:
            filename: The path of the input file.
            months_column_index: The column index of months.
            years_column_index: The column index of years.
            metrics_column_index: The column index of metrics.

        Returns:
            A tuple of numpy arrays. The first element is a
            numpy.chararray representing the timestamps and
            the second element is a numpy.ndarray representing
            the metrics.

        """

        def load_txt_by_col(col: int, data_type: str) -> np.ndarray:
            """Load data by column.

            Args:
                col: The column index, starting from 0.
                data_type: Data type of the data to be loaded.

            Returns:
                A numpy.ndarray representing the data loaded in.

            """
            return np.loadtxt(filename,
                              delimiter=",",
                              skiprows=1,
                              usecols=col,
                              dtype=data_type)

        metrics = load_txt_by_col(metrics_column_index, 'int')

        month_string_to_decimal_number = np.vectorize(lambda month: '{:%m}'.format(datetime.strptime(month, '%b')))
        months = np.char.array(
            month_string_to_decimal_number(np.array(load_txt_by_col(months_column_index, 'str')).astype('U3')))
        years = np.char.array(load_txt_by_col(years_column_index, 'str'))
        timestamps = years + '-' + months

        return timestamps, metrics

    def get_moving_average(self, window_size: int) -> None:
        """Calculate the moving average under the specified window size (m value).

        Calculate the moving average and add it to the attribute called
        moving_averages. This attribute is a dictionary, where each
        key is a window size number (m value) and each corresponding
        value is a numpy.ndarray representing the moving average under
        given window size. The empty values are set to be NaN.

        Args:
            window_size: The window size when computing moving average,
                i.e. the number of elements used to calculate one average
                each time.

        """
        raw_moving_averages = np.convolve(self.metrics, np.ones(window_size), 'valid') / window_size
        self.moving_averages[window_size] = np.append(
            np.repeat(np.nan, len(self.timestamps) - len(raw_moving_averages)),
            raw_moving_averages)

    def preprocess_before_linear_regression(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data before linear regression.

        This function does the following cleanups:
        1. Transform the timestamps to consecutive integers, and get rid of the
            values whose corresponding y values are NaN, return as x.
        2. Get rid of the NaN values of the moving average under the given
            window size, return as y.

        Args:
            window_size: The window size when computing moving average,
                i.e. the number of elements used to calculate one average
                each time.

        Returns:
            A tuple where the first element is the x data and the second
            element is the y data for linear regression.

        """
        x = np.array([i for i in range(window_size, len(self.timestamps) + 1)])
        y = self.moving_averages[window_size][window_size - 1:]
        return x, y

    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray, is_y_intercept_zero: bool = False) \
            -> Tuple[List[Union[np.float, int]], np.ndarray]:
        """Perform linear regression to the given x and y.

        Args:
            x: The data of x-axis.
            y: The data of y-axis.
            is_y_intercept_zero: True if we want the y-intercept
                of the linear regression model to be zero, False
                otherwise. The default value is False.

        Returns:
            A tuple with information of the linear regression model.
                The first element of the tuple is a list, where the
                first element is the slope and the second element is
                the y-intercept. The second element of the tuple
                is a numpy.ndarray containing the r-square.

        """
        if is_y_intercept_zero:
            x = x.reshape(-1, 1)
            betas, r_square, _, _ = np.linalg.lstsq(x, y, rcond=None)
            return [betas[0], 0], r_square
        else:
            x = np.vstack([np.ones(len(x)), x]).T
            betas, r_square, _, _ = np.linalg.lstsq(x, y, rcond=None)
            return [betas[1], betas[0]], r_square

    def predict_y_hat(self, slope: np.float, y_intercept: Union[np.float, int]) -> np.ndarray:
        """Get predictive y values by given slope and y-intercept.

        Args:
            slope: The slope of the linear regression model.
            y_intercept: The y-intercept of the linear regression
                model.

        Returns:
            A numpy.ndarray of the predictive y values.

        """
        x_series = np.array([j for j in range(1, len(self.timestamps) + 1)]).reshape(-1, 1)
        return x_series * slope + y_intercept

    def predict_data_point(self,
                           slope: np.float,
                           y_intercept: Union[np.float, int],
                           further_time_steps: int) -> np.float:
        """Get a predictive point.

        Args:
            slope: The slope of the linear regression model.
            y_intercept: The y-intercept of the linear regression
                model.
            further_time_steps: The number of steps after the last
                timestamp of the original data. For example, the last
                timestamp of the original data is 2014-12, then
                2015-01 has the further_time_steps of 1, 2015-02 has
                the further_time_steps of 2, 2015-05 has the further_
                time_steps of 5. Since the last y value of the original
                data is y_{n-1}, then the y_{n} should have further_time_
                steps of 1, y_{n+x} should have further_time_steps of x+1.

        Returns:
            The predictive y value of a data point.

        """
        x_point = len(self.timestamps) + further_time_steps
        return x_point * slope + y_intercept

    @staticmethod
    def line_plot(x_data: np.chararray,
                  y_data: np.ndarray,
                  x_axis_label: str,
                  y_axis_label: str,
                  y_axis_legend_label: str,
                  plot_title: str,
                  other_y_axis: List[Tuple[str, np.ndarray]] = None) -> None:
        """Plot line chart with the given data.

        Args:
            x_data: The data of x-axis.
            y_data: The data of y-axis.
            x_axis_label: The label of x-axis.
            y_axis_label: The label of y-axis.
            y_axis_legend_label: The legend label of y-axis.
            plot_title: The title of the plot.
            other_y_axis: If the plot does not contain multiple
                lines, this value is None(by default). Pass in this
                value if the plot need to include multiple lines.
                This is a list of tuples where each tuple's
                first element being the legend label of a y value
                and second element being the data of y value.

        """
        plt.figure(figsize=(20, 8))
        plt.xticks(rotation=45, fontsize=7)

        plt.plot(x_data, y_data, label=y_axis_legend_label)
        if other_y_axis:
            for other_y in other_y_axis:
                plt.plot(x_data, other_y[1], label=other_y[0])

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.title(plot_title)

        plt.legend()
        plt.show()
