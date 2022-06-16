import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from forecast.metrics import mean_squared_error, mean_absolute_percentage_error

class MA:
    
    """
    Moving Average model with anomaly detection using Brutlag method
    
    ### Parameters:
    
    * ts : pandas.Series
        * time series
    * from_window : int 
        * initial value of a range of MA window periods to be tested
    * to_window : int
        * final value of a range of MA window periods to be tested
    * eval_method : {default: 'rmse', 'mape'}, str
        * error metric used to find best parameters
    * allow_negative_hist : bool
        * if False, replace all negative values with 0
    * allow_negative_preds : bool
        * if False, doesn't allow any negative prediction
    * verbose : boolean
        * if True, some status messages are shown
    
    """
    
    def __init__(self, ts : pd.Series, from_window : int = 2, to_window : int = 8, eval_method : str = 'rmse', allow_negative_hist : bool = False, verbose : bool = False) -> None:
        self.ts = np.array(ts)
        self.from_window = from_window
        self.to_window = to_window
        self.eval_method = eval_method
        self.allow_negative_hist = allow_negative_hist
        self.best_score = np.inf
        self.best_window = 1
        self.real_preds = []
        self.preds_vals = []
        self.verbose = verbose

    def moving_average(self, len_preds : int = 12) -> np.array:
        
        """
        Make predictions using Moving Average with best found parameter
        
        ### Parameters:
        1. len_preds : int
            \t- prediction horizon
            
        ### Returns:
        1. real_preds : numpy.array
            \t- array containing historical and future predictions
        """

        # replacing negative values for 0
        if not self.allow_negative_hist:
            self.ts = np.where(self.ts < 0, 0, self.ts)

        # initializing some variables 
        best_preds = []
        best_hist = []
        self.best_window = 1
        actual_score = np.inf
        self.best_score = np.inf

        # looking for the best window period for moving average
        for window in range(self.from_window, self.to_window + 1):

            try:
                # initializing some variables
                preds = []
                hist = []

                # iterating through time series
                for i in range( len(self.ts[ window - 1 : ]) ):
                    preds.append( np.mean(self.ts[ i : window + i ]) )
                    hist.append( self.ts[ window + i - 1 ] )

                # calculating actual score
                if self.eval_method == 'rmse':
                    actual_score = np.sqrt(mean_squared_error(hist, preds))
                elif self.eval_method == 'mape':
                    actual_score = mean_absolute_percentage_error(hist, preds)
                else:
                    raise Exception("Invalid error metric. Two options available: 'rmse' and 'mape'")

                # if actual score is better than best score, the new window is chosen
                if actual_score < self.best_score:
                    self.best_score = actual_score
                    best_preds = preds
                    best_hist = hist
                    self.best_window = window
                    
            except:
                pass

        preds = np.array(best_preds)
        hist = np.array(best_hist)

        # printing best score and best window
        if self.verbose:
            print(f'Best score: {self.best_score} - Best window: {self.best_window}')

        hist_preds = list(self.ts)
        self.preds_vals = []

        # predicting the future
        for i in range(len_preds):
            # calculating next prediction
            pred_val = np.mean(hist_preds[ -self.best_window : ])
            # apendding prediction in history to predict next value
            hist_preds = np.append(hist_preds, pred_val)
            # saving the prediction
            self.preds_vals.append(pred_val)

        # building an array with all predictions, replacing for np.nan the first values used for the first prediction
        self.real_preds = np.concatenate((np.array( (self.best_window) * [np.nan] ), preds, self.preds_vals))

        # returning all the predictions
        return self.real_preds

    def plot_moving_average(self) -> None:

        """
        Plot original time series, Moving Average fitted time series and predictions
        """

        # replacing negative values for 0
        if not self.allow_negative_hist:
            self.ts = np.where(self.ts < 0, 0, self.ts)

        # plotting ts and real_preds
        plt.figure(figsize=(20, 10))
        plt.plot(self.real_preds, label="Model")
        plt.plot(self.ts, label="Actual")

        # getting best_score (error metric)
        error = self.best_score

        # define the chart title according to error metric
        if self.eval_method == 'mape':
            plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
        elif self.eval_method == 'rmse':
            plt.title("Root Mean Squared Error: {0:.2f}".format(error))

        # concatenate all values (self.ts and predictions) to get min and max
        all_values = np.array(list(self.ts) + list(self.real_preds))

        # plot vertical line where history and predictions are divided
        plt.vlines(
            len(self.ts),
            ymin=min(all_values),
            ymax=max(all_values),
            linestyles="dashed",
        )

        # some chart's layout adjustments
        plt.axvspan(len(self.ts) - 1, len(self.real_preds), alpha=0.3, color="lightgrey")
        plt.grid(True)
        plt.axis("tight")
        plt.legend(loc="best", fontsize=13)