import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

from forecast.holt_winters import HoltWinters
from forecast.moving_average import MA
from forecast.preprocessing import create_series_one_product_ready_to_prediction, create_series_product_store_ready_to_prediction


def predict_for_one_sku(ts : pd.Series, len_preds : int = 12, slens : list = [4, 12, 16, 26, 52], from_window_ma : int = 2, 
                         to_window_ma : int = 8, plot : bool = False, method : str = 'all', eval_method : str = 'rmse', 
                         allow_negative_hist : bool = False, allow_negative_preds: bool = False, verbose : bool = False, 
                         historical_preds : list = None) -> list:
    
    """
    Predict for one SKU using Triple Exponential Smoothing (TES) + Moving Average (MA) 
    and selecting best parameters evaluating error metric
    
    ### Parameters:
    
        * ts : pandas.Series
            * initial time series
        * len_preds : int
            * prediction horizon
        * from_window : int 
            * initial value of a range of window periods to be tested
        * to_window : int
            * final value of a range of window periods to be tested
        * slens : list
            * list of length of season
        * plot : list 
            * if True, plot a chart of   time series and predictions
        * method : {default: 'all', 'tes', 'ma'}, str
            * forecasting method
        * eval_method : {default: 'rmse', 'mape'}, str
            * error metric used to find best parameters
        * allow_negative_hist : bool
            * if False, replace all negative sales values with 0
        * allow_negative_preds : bool
            * if False, doesn't allow any negative prediction
        * verbose : bool
            * if True, some status messages are shown
        * historical_preds : list
            * list passed as reference where all historical predictions will be stored
        
    ### Returns:
    
        * results : list
            * a list with size len_preds + 7 containing best method, best error score, best parameters (alpha, beta, gama, slen, window_ma) and all predictions
    """
    
    ts = np.array(ts)
    
    if not allow_negative_hist:
        ts = np.where(ts < 0, 0, ts)
        
    ########## TRIPLE EXPONENTIAL SMOOTHING ##########
    
    if method == 'tes' or method == 'all':
        
        alpha_vals, beta_vals, gamma_vals = (np.round(np.arange(0, 1, .1), 2) for i in range(3))
        slen_vals = np.array(slens)
        best_score_tes = np.inf

        if len(ts) > 8:
            best_params = (0.5, 0.5, 0.5, 4)
        else:
            best_params = (0.5, 0.5, 0.5, 1)

        for i in product(alpha_vals, beta_vals, gamma_vals, slen_vals):
            try:

                alpha_i, beta_i, gamma_i, slen_i = i
                model_tes = HoltWinters(ts, slen=slen_i, alpha=alpha_i, beta=beta_i, gamma=gamma_i, n_preds=len_preds)
                model_tes.triple_exponential_smoothing()

                if eval_method == 'mape':
                    actual_score = model_tes.calculate_mean_absolute_percentage_error()
                elif eval_method == 'rmse':
                    actual_score = model_tes.calculate_root_mean_squared_error()

                if allow_negative_preds:
                    if actual_score < best_score_tes:
                        best_score_tes = actual_score
                        best_params = i
                else:
                    if (actual_score < best_score_tes) and (np.count_nonzero( np.array(model_tes.result[-len_preds : ]) < 0 ) <= 0):
                        best_score_tes = actual_score
                        best_params = i

            except:
    #             print('Ocorreu um erro')
                pass

        try:

            alpha_final, beta_final, gamma_final, slen_final = best_params

            model_tes = HoltWinters(series = ts, 
                                slen = slen_final, 
                                alpha=alpha_final, 
                                beta=beta_final, 
                                gamma=gamma_final, 
                                n_preds=len_preds)

            model_tes.triple_exponential_smoothing()

        except:
            print(f'Ocorreu um erro // Best params: {best_params} // Best score: {best_score_tes} // TS size: {len(ts)}')

        if verbose:
            print(f'Best params: {best_params} - Best score: {best_score_tes}')
            print(f'Predictions: {model_tes.result[-len_preds:]}')
            
        if plot:
            model_tes.plotHoltWinters(plot_anomalies=True, plot_intervals=True, known_error=best_score_tes, eval_method_title=eval_method)
           
        if method == 'tes':
            best_method = 'TES'
            
            if historical_preds is not None:
                historical_preds.append(model_tes.result[:-len_preds])
            
            return [best_method, best_score_tes] + list(best_params) + [None] + model_tes.result[-len_preds:]
        
    ########## MOVING AVERAGE ##########
    
    if method == 'ma' or method == 'all':

        model_ma = MA(ts=ts, from_window=from_window_ma, to_window=to_window_ma, eval_method=eval_method, allow_negative_hist=allow_negative_hist, verbose=verbose)
        model_ma.moving_average(len_preds=len_preds)

        if plot:
            model_ma.plot_moving_average()

        if method == 'ma':
            best_method = 'MA'
            
            if historical_preds is not None:
                historical_preds.append(model_ma.real_preds[:-(len_preds+1)])
            
            return [best_method, model_ma.best_score] + [None, None, None, None] + [model_ma.best_window] + model_ma.preds_vals

    best_method = 'MA' if model_ma.best_score <= best_score_tes else 'TES'
    
    if best_method == 'MA':
        
        if historical_preds is not None:
                historical_preds.append(model_ma.real_preds[:-(len_preds+1)])
        
        # score + tes_params + ma_window + predictions
        return [best_method, model_ma.best_score] + list(best_params) + [model_ma.best_window] + model_ma.preds_vals
    else:
        
        if historical_preds is not None:
                historical_preds.append(model_tes.result[:-len_preds])
        
        return [best_method, best_score_tes] + list(best_params) + [model_ma.best_window] + model_tes.result[-len_preds:]


def predict_many_skus(df : pd.DataFrame, list_of_skus : list, current_wk : int, len_preds : int = 12, slens : list = [4, 12, 16, 26, 52], 
                       from_window_ma : int = 2, to_window_ma : int = 8, plot : bool = False, method : str = 'all', eval_method : str = 'rmse', 
                       allow_negative_hist : bool = False, allow_negative_preds : bool = False, use_tqdm : bool = True, verbose : bool = False, 
                       historical_preds : list = None) -> pd.DataFrame:
    
    """
    Predict for many SKUs using Triple Exponential Smoothing (TES) + Moving Average (MA) 
    and selecting best parameters evaluating error metric

    ### Parameters:
    
        * df : pandas.DataFrame
            * initial time series
        * list_of_skus : list
            * list containing all SKUs to be predicted
        * current_wk : int
            * current week number in format %Y%W (e.g: 202132)
        * len_preds : int
            * prediction horizon
        * from_window : int 
            * initial value of a range of window periods to be tested
        * to_window : int
            * final value of a range of window periods to be tested
        * slens : list
            * list of length of season
        * plot : list 
            * if True, plot a chart of   time series and predictions
        * method : {default: 'all', 'tes', 'ma'}, str
            * forecasting method
        * eval_method : {default: 'rmse', 'mape'}, str
            * error metric used to find best parameters
        * allow_negative_hist : bool
            * if False, replace all negative sales values with 0
        * allow_negative_preds : bool
            * if False, doesn't allow any negative prediction
        * verbose : bool
            * if True, some status messages are shown
        * historical_preds : list
            * list passed as reference where all historical predictions will be stored
        
    ### Returns:
    
        * results : pandas.DataFrame
            * dataframe with len_preds + 7 columns containing best method, best error score, best parameters (alpha, beta, gama, slen, window_ma) and all predictions 
    """
    
    all_results = []
    col_names = ['Best_Method', 'RMSE', 'Alpha', 'Beta', 'Gamma', 'Season_Length', 'MA_Window'] + [f"W+{i}" for i in range(len_preds)]
    skus_idx = []
    n_skus = len(list_of_skus)
    
    if use_tqdm:
        for sku in tqdm(list_of_skus):
            if verbose:
                print(f'SKU: {sku}')
            skus_idx.append(sku)
            ts = create_series_one_product_ready_to_prediction(df, sku, current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
    else:
        for enum, sku in enumerate(list_of_skus):
            if verbose:
                print(f'SKU: {sku} - Status: {enum + 1}/{n_skus}')
            skus_idx.append(sku)
            ts = create_series_one_product_ready_to_prediction(df, sku, current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
        
    df_results = pd.DataFrame(all_results, columns=col_names, index=skus_idx).reset_index().rename(columns={'index': 'SKU'})
        
    return df_results

def predict_many_skus_based_on_dataframe(df : pd.DataFrame, current_wk : int, sku_column : str = 'Product', len_preds : int = 12, slens : list = [4, 12, 16, 26, 52], 
                                         from_window_ma : int = 2, to_window_ma : int = 8, plot : bool = False, method : str = 'all', eval_method : str = 'rmse', 
                                         allow_negative_hist : bool = False, allow_negative_preds : bool = False, use_tqdm : bool = True, verbose : bool = False, 
                                         historical_preds : list = None) -> pd.DataFrame:
     
    """
    Predict for many SKUs stored in a dataframe's column using Triple Exponential Smoothing (TES) + Moving Average (MA) 
    and selecting best parameters evaluating error metric

    ### Parameters:
    
        * df : pandas.DataFrame
            * initial time series
        * current_wk : int
            * current week number in format %Y%W (e.g: 202132)
        * sku_column : str
            * name of the column which contains the SKUs
        * len_preds : int
            * prediction horizon
        * from_window : int 
            * initial value of a range of window periods to be tested
        * to_window : int
            * final value of a range of window periods to be tested
        * slens : list
            * list of length of season
        * plot : list 
            * if True, plot a chart of   time series and predictions
        * method : {default: 'all', 'tes', 'ma'}, str
            * forecasting method
        * eval_method : {default: 'rmse', 'mape'}, str
            * error metric used to find best parameters
        * allow_negative_hist : bool
            * if False, replace all negative sales values with 0
        * allow_negative_preds : bool
            * if False, doesn't allow any negative prediction
        * verbose : bool
            * if True, some status messages are shown
        * historical_preds : list
            * list passed as reference where all historical predictions will be stored
        
    ### Returns:
    
        * results : pandas.DataFrame
            * dataframe with len_preds + 7 columns containing best method, best error score, best parameters (alpha, beta, gama, slen, window_ma) and all predictions
    """
    
    list_of_skus = df[sku_column].unique()
    
    all_results = []
    col_names = ['Best_Method', 'RMSE', 'Alpha', 'Beta', 'Gamma', 'Season_Length', 'MA_Window'] + [f"W+{i}" for i in range(len_preds)]
    skus_idx = []
    n_skus = len(list_of_skus)
    
    if use_tqdm:
        for sku in tqdm(list_of_skus):
            if verbose:
                print(f'SKU: {sku}')
            skus_idx.append(sku)
            ts = create_series_one_product_ready_to_prediction(df, sku, current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
    else:
        for enum, sku in enumerate(list_of_skus):
            if verbose:
                print(f'SKU: {sku} - Status: {enum + 1}/{n_skus}')
            skus_idx.append(sku)
            ts = create_series_one_product_ready_to_prediction(df, sku, current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
        
    df_results = pd.DataFrame(all_results, columns=col_names, index=skus_idx).reset_index().rename(columns={'index': 'SKU'})
        
    return df_results

def predict_many_sku_stores(df : pd.DataFrame, list_of_sku_stores : list, current_wk : int, len_preds : int = 12, slens : list = [4, 12, 16, 26, 52], 
                           from_window_ma : int = 2, to_window_ma : int = 8, plot : bool = False, method : str = 'all', eval_method : str = 'rmse', 
                           allow_negative_hist : bool = False, allow_negative_preds : bool = False, use_tqdm : bool = True, verbose : bool =False, 
                           historical_preds : list = None) -> pd.DataFrame:
    
    """
    ### Parameters:
    
        * df : pandas.DataFrame
            * initial time series
        * list_of_sku_stores : list
            * list containing all combinations SKU-Stores to be predicted
        * current_wk : int
            * current week number in format %Y%W (e.g: 202132)
        * len_preds : int
            * prediction horizon
        * from_window : int 
            * initial value of a range of window periods to be tested
        * to_window : int
            * final value of a range of window periods to be tested
        * slens : list
            * list of length of season
        * plot : list 
            * if True, plot a chart of   time series and predictions
        * method : {default: 'all', 'tes', 'ma'}, str
            * forecasting method
        * eval_method : {default: 'rmse', 'mape'}, str
            * error metric used to find best parameters
        * allow_negative_hist : bool
            * if False, replace all negative sales values with 0
        * allow_negative_preds : bool
            * if False, doesn't allow any negative prediction
        * verbose : bool
            * if True, some status messages are shown
        * historical_preds : list
            * list passed as reference where all historical predictions will be stored
        
    ### Returns:
    
        * results : pandas.DataFrame
            * dataframe with len_preds + 7 columns containing best method, best error score, best parameters (alpha, beta, gama, slen, window_ma) and all predictions
    """
    
    all_results = []
    col_names = ['Best_Method', 'RMSE', 'Alpha', 'Beta', 'Gamma', 'Season_Length', 'MA_Window'] + [f"W+{i}" for i in range(len_preds)]
    skus_idx = []
    list_of_sku_stores = list(list_of_sku_stores)
    n_skus = len(list_of_sku_stores)
    
    if use_tqdm:
        for sku_store in tqdm(list_of_sku_stores):
            if verbose:
                print(f'SKU-Store: {sku_store}')
            skus_idx.append(sku_store)
            ts = create_series_product_store_ready_to_prediction(df, sku_store[0], sku_store[1], current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
    else:
        for enum, sku_store in enumerate(list_of_sku_stores):
            if verbose:
                print(f'SKU-Store: {sku_store} - Status: {enum + 1}/{n_skus}')
            skus_idx.append(sku_store)
            ts = create_series_product_store_ready_to_prediction(df, sku_store[0], sku_store[1], current_wk=current_wk)
            results = predict_for_one_sku(ts.values, len_preds=len_preds, slens=slens, from_window_ma=from_window_ma, to_window_ma=to_window_ma, plot=plot, method=method, eval_method=eval_method, allow_negative_hist=allow_negative_hist, allow_negative_preds=allow_negative_preds, verbose=verbose, historical_preds=historical_preds)
            all_results.append(results)
            
    m_index = pd.MultiIndex.from_tuples(skus_idx, names=['SKU', 'Store'])
        
    df_results = pd.DataFrame(all_results, columns=col_names, index=m_index).reset_index()
        
    return df_results