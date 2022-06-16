from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from forecast.metrics import mean_squared_error, mean_absolute_percentage_error

class HoltWinters:

    """
    Holt-Winters model with the anomalies detection using Brutlag method
    
    ### Parameters:
    
        * series : pandas.Series 
            * initial time series
        * slen : int
            * length of a season
        * alpha: float
            * Holt-Winters model's alpha coefficient
        * beta : float
            * Holt-Winters model's beta coefficient
        * gamma : float
            * Holt-Winters model's gamma coefficient
        * n_preds : int 
            * predictions horizon
        * scaling_factor : float 
            * sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    """

    def __init__(self, series : pd.Series, slen : int, alpha : float, beta : float, gamma : float, n_preds : int, scaling_factor : float = 1.96) -> None:
        self.series = np.array(series)
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        self.best_score = np.inf
        self.best_params = None

    def initial_trend(self) -> float:

        """
        Compute initial trend for Holt-Winters model
        """

        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self) -> dict:

        """
        Compute initial seasonal components for Holt-Winters model
        """
        
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)

        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(
                sum(self.series[self.slen * j : self.slen * j + self.slen]) / float(self.slen)
            )

        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (
                    self.series[self.slen * j + i] - season_averages[j]
                )
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        
        """
        Make predictions using TES
        """
        
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(
                    self.result[0] + self.scaling_factor * self.PredictedDeviation[0]
                )

                self.LowerBond.append(
                    self.result[0] - self.scaling_factor * self.PredictedDeviation[0]
                )
                continue

            if i >= len(self.series):  # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = (
                    smooth,
                    self.alpha * (val - seasonals[i % self.slen])
                    + (1 - self.alpha) * (smooth + trend),
                )
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (
                    self.gamma * (val - smooth)
                    + (1 - self.gamma) * seasonals[i % self.slen]
                )
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(
                    self.gamma * np.abs(self.series[i] - self.result[i])
                    + (1 - self.gamma) * self.PredictedDeviation[-1]
                )

            self.UpperBond.append(
                self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.LowerBond.append(
                self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1]
            )

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

    def calculate_mean_absolute_percentage_error(self):
        return mean_absolute_percentage_error(self.series, self.result[: len(self.series)])

    def calculate_root_mean_squared_error(self):
        return np.sqrt(mean_squared_error(self.series, self.result[: len(self.series)]))
    
    def plotHoltWinters(self, plot_intervals : bool = False, plot_anomalies : bool = False, known_error : Union[None, float] = None, eval_method_title : str = 'rmse'):
        
        """
        Plot original time series, Holt-Winters fitted time series and predictions
        
        ### Parameters:
        
        * series : pandas.Series
            *dataset with timeseries
        * plot_intervals : bool
            * show confidence intervals
        * plot_anomalies : bool
            * show anomalies
        * known_error : float 
            * if not None, it is plotted in the chart as the error value
        * eval_method_title : str
            * define the error metric title based on the error metric choosen
        """

        plt.figure(figsize=(20, 10))
        plt.plot(self.result, label="Model")
        plt.plot(self.series, label="Actual")
        
        if known_error == None:
            if eval_method_title == 'mape':
                error = mean_absolute_percentage_error(self.series, self.result[: len(self.series)])
            elif eval_method_title == 'rmse':
                error = np.sqrt(mean_squared_error(self.series, self.result[: len(self.series)]))
        else:
            error = known_error
           
        if eval_method_title == 'mape':
            plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        elif eval_method_title == 'rmse':
            plt.title("Root Mean Squared Error: {0:.2f}".format(error))

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(self.series))
            anomalies[self.series < self.LowerBond[: len(self.series)]] = self.series[
                self.series < self.LowerBond[: len(self.series)]
            ]
            anomalies[self.series > self.UpperBond[: len(self.series)]] = self.series[
                self.series > self.UpperBond[: len(self.series)]
            ]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

        if plot_intervals:
            plt.plot(self.UpperBond, "r--", alpha=0.5, label="Up/Low confidence")
            plt.plot(self.LowerBond, "r--", alpha=0.5)
            plt.fill_between(
                x=range(0, len(self.result)),
                y1=self.UpperBond,
                y2=self.LowerBond,
                alpha=0.2,
                color="grey",
            )

        plt.vlines(
            len(self.series),
            ymin=min(self.LowerBond),
            ymax=max(self.UpperBond),
            linestyles="dashed",
        )
        plt.axvspan(len(self.series) - 1, len(self.result), alpha=0.3, color="lightgrey")
        plt.grid(True)
        plt.axis("tight")
        plt.legend(loc="best", fontsize=13)
        
def predict_tes_for_one_product(ts : pd.Series, len_preds : int = 12, slens : list = [4, 12, 16, 26, 52], plot : bool = False, eval_method : str = 'rmse', 
                                allow_negative_hist : bool = False, allow_negative_preds : bool = False, verbose : bool = False) -> list :
    
    """
    Predict for one product using Triple Exponential Smoothing (TES)
    
    ### Parameters:

    * ts : pandas.Series
        * initial time series
    * len_preds : int
        * prediction horizon
    * slens : list 
        * list of length of season
    * plot : bool
        * if True, plot a chart of time series and predictions
    * eval_method : {'rmse', 'mape'}, str
        * error metric used to find best parameters
    * allow_negative_hist : bool
        * if False, replace all negative sales values with 0
    * allow_negative_preds : bool
        * if False, doesn't allow any negative prediction
    * verbose : bool
        * if True, some status messages are shown
        
    ### Returns:

    * results : list
        * a list with len_preds+1 size containing the best error score and all the predictions
    """
    
    ts = np.array(ts)
    
    if not allow_negative_hist:
        ts = np.where(ts < 0, 0, ts)
        
    alpha_vals, beta_vals, gamma_vals = (np.round(np.arange(0, 1, .1), 2) for i in range(3))
    slen_vals = np.array(slens)
    best_score = np.inf
#     best_params = None
    if len(ts) > 8:
        best_params = (0.5, 0.5, 0.5, 4)
    else:
        best_params = (0.5, 0.5, 0.5, 1)

    for i in product(alpha_vals, beta_vals, gamma_vals, slen_vals):
        try:
            
            alpha_i, beta_i, gamma_i, slen_i = i
            model = HoltWinters(ts, slen=slen_i, alpha=alpha_i, beta=beta_i, gamma=gamma_i, n_preds=len_preds)
            model.triple_exponential_smoothing()
            
            if eval_method == 'mape':
                actual_score = model.calculate_mean_absolute_percentage_error()
            elif eval_method == 'rmse':
                actual_score = model.calculate_root_mean_squared_error()
                
            if allow_negative_preds:
                if actual_score < best_score:
                    best_score = actual_score
                    best_params = i
            else:
                if (actual_score < best_score) and (np.count_nonzero( np.array(model.result[-len_preds : ]) < 0 ) <= 0):
                    best_score = actual_score
                    best_params = i
            
        except:
#             print('Ocorreu um erro')
            pass

    try:

        alpha_final, beta_final, gamma_final, slen_final = best_params

        model = HoltWinters(series = ts, 
                            slen = slen_final, 
                            alpha=alpha_final, 
                            beta=beta_final, 
                            gamma=gamma_final, 
                            n_preds=len_preds)

        model.triple_exponential_smoothing()
    
    except:
        print(f'Ocorreu um erro // Best params: {best_params} // Best score: {best_score} // TS size: {len(ts)}')

    if verbose:
        print(f'Best params: {best_params} - Best score: {best_score}')
        print(f'Predictions: {model.result[-len_preds:]}')
    
    if plot:
        model.plotHoltWinters(plot_anomalies=True, plot_intervals=True, known_error=best_score, eval_method_title=eval_method)
        
    return [best_score] + model.result[-len_preds:]
        
def predict_tes_for_one_product_with_train_test_split(ts : pd.Series, len_preds : int = 12, len_test : int = 12, share_test : float = 0.5, slens : list = [4, 12, 16, 26], 
                                                      plot : bool = False, eval_method : str = 'rmse', allow_negative_hist : bool = False, 
                                                      allow_negative_preds : bool = False, verbose : bool = False) -> list:
    
    """
    Predict for one Product using Triple Exponential Smoothing (TES) and select best parameters comparing real predictions to real values
    
    ### Parameters:

    * ts : pandas.Series
        * initial time series
    * len_preds : int
        * prediction horizon
    * len_test : int
        * size of test dataset used to evaluate parameters (deprecated)
    * share_test : float
        * share of original time series segregated to evaluate parameters
    * slens : list
        * list of length of season
    * plot : bool
        * if True, plot a chart of time series and predictions
    * eval_method : {'rmse', 'mape'}, str
        * error metric used to find best parameters
    * allow_negative_hist : bool
        * if False, replace all negative sales values with 0
    * allow_negative_preds : bool
        * if False, doesn't allow any negative prediction
    * verbose : bool
        * if True, some status messages are shown
        
    ### Returns:

    * results : list
        * a list with size len_preds+1 containing best error score and all predictions
    """
    
    ts = np.array(ts)
    
    if not allow_negative_hist:
        ts = np.where(ts < 0, 0, ts)
    
    alpha_vals, beta_vals, gamma_vals = (np.round(np.arange(0, 1, .1), 2) for i in range(3))
    slen_vals = np.array(slens)
    best_score = np.inf
    actual_score = np.inf
#     best_params = None
    if len(ts) > 8:
        best_params = (0.5, 0.5, 0.5, 4)
    else:
        best_params = (0.5, 0.5, 0.5, 1)
    
    len_test = int(share_test * len(ts))
    ts_train = ts[ : -len_test]
    ts_test = ts[-len_test : ]

    for i in product(alpha_vals, beta_vals, gamma_vals, slen_vals):
        try:
            
            alpha_i, beta_i, gamma_i, slen_i = i
            model = HoltWinters(ts, slen=slen_i, alpha=alpha_i, beta=beta_i, gamma=gamma_i, n_preds=len_preds)
            model.triple_exponential_smoothing()
            
            if eval_method == 'mape':
                actual_score = model.calculate_mean_absolute_percentage_error()
            elif eval_method == 'rmse':
                actual_score = model.calculate_root_mean_squared_error()
                
            if allow_negative_preds:
                if actual_score < best_score:
                    best_score = actual_score
                    best_params = i
            else:
                if (actual_score < best_score) and (np.count_nonzero( np.array(model.result[-len_preds : ]) < 0 ) <= 0):
                    best_score = actual_score
                    best_params = i
            
        except:
#             print('Ocorreu um erro')
            pass

    try:

        alpha_final, beta_final, gamma_final, slen_final = best_params

        model = HoltWinters(series = ts, 
                            slen = slen_final, 
                            alpha=alpha_final, 
                            beta=beta_final, 
                            gamma=gamma_final, 
                            n_preds=len_preds)

        model.triple_exponential_smoothing()
    
    except:
        print(f'Ocorreu um erro // Best params: {best_params} // Best score: {best_score} // TS size: {len(ts)}')

    if verbose:
        print(f'Best params: {best_params} - Best score: {best_score}')
        print(f'Predictions: {model.result[-len_preds:]}')
    
    if plot:
        model.plotHoltWinters(plot_anomalies=True, plot_intervals=True, known_error=best_score, eval_method_title=eval_method)
        
    return [best_score] + model.result[-len_preds:]