from typing import Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true : Union[list, np.ndarray], y_pred : Union[list, np.ndarray]) -> float:
    
    """
    Compute Mean Absolute Percentage Error
    
    ### Parameters:
    
        * y_true : list, numpy.array 
            * real values
        * y_pred : list, numpy.array 
            * predicted values

    ### Returns

        * mape : float
            * mean_absolute_percentage_error
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Operations over DataFrames

def calc_forecast_error_ma4(g):
    
    rmse = np.sqrt(mean_squared_error( g['SaleQty'], g['Forecast_MA4'] ))
    mape = mean_absolute_percentage_error( g['SaleQty'], g['Forecast_MA4'] )
    mae = mean_absolute_error( g['SaleQty'], g['Forecast_MA4'] )
    mae_perc = mean_absolute_error( g['SaleQty'], g['Forecast_MA4'] ) / np.mean( g['SaleQty'] )
    sum_sls = g['SaleQty'].sum()
    sum_frcst = g['Forecast_MA4'].sum()
    bias = sum_frcst / sum_sls
    
    return pd.Series({ 'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'MAE%': mae_perc, 'Bias': bias, 'SaleQty': sum_sls, 'Forecast': sum_frcst })

def calc_forecast_error_tes(g):
    
    rmse = np.sqrt(mean_squared_error( g['SaleQty'], g['Forecast_TES'] ))
    mape = mean_absolute_percentage_error( g['SaleQty'], g['Forecast_TES'] )
    mae = mean_absolute_error( g['SaleQty'], g['Forecast_TES'] )
    mae_perc = mean_absolute_error( g['SaleQty'], g['Forecast_TES'] ) / np.mean( g['SaleQty'] )
    sum_sls = g['SaleQty'].sum()
    sum_frcst = g['Forecast_TES'].sum()
    bias = sum_frcst / sum_sls
    
    return pd.Series({ 'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'MAE%': mae_perc, 'Bias': bias, 'SaleQty': sum_sls, 'Forecast': sum_frcst  })