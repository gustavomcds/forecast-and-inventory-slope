from typing import Union
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.stats import linregress

def calc_pqr(df : pd.DataFrame, sku_id_col : str = 'Product', sls_qty_col : str = 'SaleQty', 
             dt_col : str = 'Date', period_in_days : int = 90) -> pd.DataFrame:
    
    """
    Compute PQR product segmentation, sorting sales values (qty only) from biggest to lowest, 
    computing the cumulative percentage and classifying according to the following rules:
    
        P: <= 80%
        Q: <= 95%
        R: > 95%
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id_col : str
            * product column name
        * sls_qty_col : str
            * sales quantity column name
        * dt_col : str
            * sales date column name
        * period_in_days : int
            * period, in days, in which sales will be evaluated to perform product segmentation
        
    ### Returns:
    
        * df_abc_pqr_by_model : pandas.DataFrame
            * segmentation ABC x PQR of each Product
        * df_abc_pqr_matrix : pandas.DataFrame
            * matrix containing the number of Products of each combination ABC x PQR
    """
     
    # defining cut date
    from_dt = datetime.strftime(datetime.now() - timedelta(days=period_in_days), format='%Y-%m-%d')
    
    # filtering dataframe from desired date
    df_sls = df[df[dt_col] >= from_dt]
    
    # selecting columns
    sls_big_qty_cols = [sku_id_col, sls_qty_col]

    # filtering dataframe
    df_sls_big_qty = df_sls[sls_big_qty_cols]

    # agrupar o dataframe por modelo + descrição e somar as vendas
    df_sls_grpd_by_model_qty = df_sls_big_qty.groupby(by=sku_id_col).sum()

    # resetting index
    df_sls_grpd_by_model_qty = df_sls_grpd_by_model_qty.reset_index()

    # excluding negative summed up sales
    df_sls_grpd_by_model_qty = df_sls_grpd_by_model_qty[df_sls_grpd_by_model_qty[sls_qty_col] >= 0]

    # ordenar da maior para a menor venda
    df_sls_grpd_by_model_qty.sort_values(by=sls_qty_col, ascending=False, inplace=True)

    # summing up sales by model in 2021
    sls_qty_sum = df_sls_grpd_by_model_qty[sls_qty_col].sum()

    # computing cumulative sum
    df_sls_grpd_by_model_qty['Cum. Sum'] = df_sls_grpd_by_model_qty[sls_qty_col].cumsum() / sls_qty_sum

    # PQR classification
    df_sls_grpd_by_model_qty['PQR'] = df_sls_grpd_by_model_qty['Cum. Sum'].apply(lambda x: 'P' if x <= 0.8 else ('Q' if x <= 0.95 else 'R'))

    # reordering columns
    df_pqr_by_model = df_pqr_by_model.reindex(columns=[sku_id_col, 'PQR'])

    # construir matriz ABC x PQR
    df_pqr_matrix = df_pqr_by_model[['PQR']].value_counts().unstack()

    # returning results
    return df_pqr_by_model, df_pqr_matrix

def calc_abc_pqr(df : pd.DataFrame, sku_id_col : str = 'Product', sls_qty_col : str = 'SaleQty', sls_val_col : str = 'SaleValue', 
                 dt_col : str = 'Date', period_in_days : int = 90) -> pd.DataFrame:
    
    """
    Compute ABC and PQR product segmentation, sorting sales values (qty and value) from biggest to lowest, 
    computing the cumulative percentage and classifying according to the following rules:
    
        A/P: <= 80%
        B/Q: <= 95%
        C/R: > 95%
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id_col : str
            * product column name
        * sls_qty_col : str
            * sales quantity column name
        * sls_val_col : str
            * sales value column name
        * dt_col : str
            * sales date column name
        * period_in_days : int
            * period, in days, in which sales will be evaluated to perform product segmentation
        
    ### Returns:
    
        * df_abc_pqr_by_model : pandas.DataFrame
            * segmentation ABC x PQR of each Product
        * df_abc_pqr_matrix : pandas.DataFrame
            * matrix containing the number of Products of each combination ABC x PQR
    """
     
    # defining cut date
    from_dt = datetime.strftime(datetime.now() - timedelta(days=period_in_days), format='%Y-%m-%d')
    
    # filtering dataframe from desired date
    df_sls = df[df[dt_col] >= from_dt]
    
    # selecting columns
    sls_big_val_cols = [sku_id_col, sls_val_col]
    sls_big_qty_cols = [sku_id_col, sls_qty_col]

    # filtering dataframe
    df_sls_big_val = df_sls[sls_big_val_cols]
    df_sls_big_qty = df_sls[sls_big_qty_cols]

    # agrupar o dataframe por modelo + descrição e somar as vendas
    df_sls_grpd_by_model_val = df_sls_big_val.groupby(by=sku_id_col).sum()
    df_sls_grpd_by_model_qty = df_sls_big_qty.groupby(by=sku_id_col).sum()

    # resetting index
    df_sls_grpd_by_model_val = df_sls_grpd_by_model_val.reset_index()
    df_sls_grpd_by_model_qty = df_sls_grpd_by_model_qty.reset_index()

    # excluding negative summed up sales
    df_sls_grpd_by_model_val = df_sls_grpd_by_model_val[df_sls_grpd_by_model_val[sls_val_col] >= 0]
    df_sls_grpd_by_model_qty = df_sls_grpd_by_model_qty[df_sls_grpd_by_model_qty[sls_qty_col] >= 0]

    # sorting from biggest to lowest
    df_sls_grpd_by_model_val.sort_values(by=sls_val_col, ascending=False, inplace=True)
    df_sls_grpd_by_model_qty.sort_values(by=sls_qty_col, ascending=False, inplace=True)

    # summing up sales by model in 2021
    sls_val_sum = df_sls_grpd_by_model_val[sls_val_col].sum()
    sls_qty_sum = df_sls_grpd_by_model_qty[sls_qty_col].sum()

    # computing cumulative sum
    df_sls_grpd_by_model_val['Cum. Sum'] = df_sls_grpd_by_model_val[sls_val_col].cumsum() / sls_val_sum
    df_sls_grpd_by_model_qty['Cum. Sum'] = df_sls_grpd_by_model_qty[sls_qty_col].cumsum() / sls_qty_sum

    # ABC and PQR classification
    df_sls_grpd_by_model_val['ABC'] = df_sls_grpd_by_model_val['Cum. Sum'].apply(lambda x: 'A' if x <= 0.8 else ('B' if x <= 0.95 else 'C'))
    df_sls_grpd_by_model_qty['PQR'] = df_sls_grpd_by_model_qty['Cum. Sum'].apply(lambda x: 'P' if x <= 0.8 else ('Q' if x <= 0.95 else 'R'))

    # merging dataframes
    df_abc_pqr_by_model = pd.merge(df_sls_grpd_by_model_val[[sku_id_col, 'ABC']], 
                                   df_sls_grpd_by_model_qty[[sku_id_col, 'PQR']], 
                                   on=sku_id_col, how='inner')

    # reordering columns
    df_abc_pqr_by_model = df_abc_pqr_by_model.reindex(columns=[sku_id_col, 'ABC', 'PQR'])

    # construir matriz ABC x PQR
    df_abc_pqr_matrix = df_abc_pqr_by_model[['ABC', 'PQR']].value_counts().unstack()

    # returning results
    return df_abc_pqr_by_model, df_abc_pqr_matrix

def calc_xyz(df : pd.DataFrame, sku_id_col : str = 'Product', sls_qty_col : str = 'SaleQty', dt_col : str = 'Date', period_in_days : int = 90, 
             wk_col : str = 'YearWeek', groupby_wk=True) -> pd.DataFrame:
    
    """
    Compute XYZ product segmentation, analysing the coefficient of variance according to following rules:
    
        X: <= 0.5
        Y: <= 0.75
        Z: > 0.75
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id_col : str
            * product column name
        * sls_qty_col : str
            * sales quantity column name
        * dt_col : str
            * sales date column name
        * period_in_days : int
            * period, in days, in which sales will be evaluated to perform product segmentation
        * wk_col : str
            * week number column name
        * groupby_wk : bool
            * if true, sales are grouped by weeks, otherwise, keep the original time level
        
    ### Returns:
    
        * df_abc_pqr_by_model : pandas.DataFrame 
            * segmentation XYZ of each Product
    """
    
    # defining cut date
    from_dt = datetime.strftime(datetime.now() - timedelta(days=period_in_days), format='%Y-%m-%d')
    
    # filtering dataframe from desired date
    df = df[df[dt_col] >= from_dt]

    # grouping dataframe by week, product id and product description
    if groupby_wk:
        df = df[[wk_col, sku_id_col, sls_qty_col]].groupby([wk_col, sku_id_col]).sum().reset_index().drop(columns=wk_col)
        df = df[df[sls_qty_col] >= 0]
        # calculating mean by product
        df_mean = df[[sku_id_col, sls_qty_col]].groupby(sku_id_col).mean().reset_index().rename(columns={sls_qty_col: 'Mean'})
        # calculating standard deviation by product
        df_std = df[[sku_id_col, sls_qty_col]].groupby(sku_id_col).std(ddof=0).reset_index().rename(columns={sls_qty_col: 'Std'})
    else:
        df = df[[sku_id_col, sls_qty_col]]
        df = df[df[sls_qty_col] >= 0]
        # calculating mean by product
        df_mean = df.groupby(sku_id_col).mean().reset_index().rename(columns={sls_qty_col: 'Mean'})#.fillna(0)
        # calculating standard deviation by product
        df_std = df.groupby(sku_id_col).std(ddof=0).reset_index().rename(columns={sls_qty_col: 'Std'})#.fillna(0)
    
    # merging dataframes
    df_coef_var = pd.merge(df_std, df_mean, on=sku_id_col, how='outer')
    
    # calculating coefficient of variation
    df_coef_var['CoefVar'] = round(df_std['Std'] / df_mean['Mean'], 2)
    # classifying products
    df_coef_var['XYZ'] = df_coef_var['CoefVar'].apply(lambda x: 'X' if x <= 0.5 else ('Y' if x<= 0.75 else 'Z'))
    
    # returning final dataframe
    return df_coef_var[[sku_id_col, 'XYZ']]

def calc_abc_pqr_xyz(df : pd.DataFrame, sku_id_col : str = 'Product', sls_qty_col : str = 'SaleQty', sls_val_col : str = 'SaleValue', 
                     dt_col : str = 'Date', wk_col : str = 'YearWeek', period_in_days : int = 90, groupby_wk : bool = True) -> pd.DataFrame:
    
    """
    Compute ABC and PQR product segmentation, sorting sales values (qt and $) from biggest to lowest, 
    computing the cumulative percentage and classifying according to following rules:
    
        A/P: <= 80%
        B/Q: <= 95%
        C/R: > 95%
    
    Also, compute XYZ product segmentation, analysing the coefficient of variance according to following rules:
    
        X: <= 0.5
        Y: <= 0.75
        Z: > 0.75
    
    Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id_col : str
            * product column name
        * sls_qty_col : str
            * sales quantity column name
        * sls_val_col : str
            * sales value column name
        * dt_col : str
            * sales date column name
        * period_in_days: int
            * period, in days, in which sales will be evaluated to perform product segmentation
        * wk_col : str
            * Week number column name
        * groupby_wk : bool
            * If true, sales are grouped by weeks, otherwise, keep the original time level
        
    Returns:
    
        * df_abc_pqr_by_model : pandas.DataFrame
            * segmentation ABC x PQR x XYZ of each Product    
    """
    
    # classifying in abc and pqr
    df_abc_pqr = calc_abc_pqr(df, sku_id_col, sls_qty_col, sls_val_col, dt_col, period_in_days)[0]
    # classifying in xyz
    df_xyz = calc_xyz(df, sku_id_col, sls_qty_col, dt_col, period_in_days, wk_col, groupby_wk)
    
    # merging dataframes
    df_abc_pqr_xyz = pd.merge(df_abc_pqr, df_xyz, on=sku_id_col, how='inner')
    
    # returning final dataframe
    return df_abc_pqr_xyz

def calculate_service_level_abc_xyz(sl_rates : list[list[float]] = [[0.97, 0.95, 0.93], [0.91, 0.90, 0.89], [0.87, 0.85, 0.80]], 
                                    sl_idx : list = ['A', 'B', 'C'], sl_cols : list = ['X', 'Y', 'Z']) -> pd.DataFrame:
    
    """
    Build a matrix of service levels for each ABC x XYZ combination
    
    ### Default matrix: 

            X | Y | Z

        A | 0.97 | 0.95 | 0.93

        B | 0.91 | 0.90 | 0.89
        
        C | 0.87 | 0.85 | 0.80
        
    ### Parameters:
    
        * sl_rates : list[list[float]]
            * 3x3 matrix containing service level for each ABC x XYZ combination
        * sl_index : list
            * pandas dataframe's index values
        * sl_cols : list
            * pandas dataframe's columns values
    
    ### Returns:
        
        * df_sl_rates : pandas.DataFrame
            * 3x3 dataframe matrix containing service level for each ABC x XYZ combination
    """
    
    return pd.DataFrame(sl_rates, columns=sl_cols, index=sl_idx)

def create_list_of_weeks(df : pd.DataFrame, current_wk : int, max_wk : int = 53, len_preds : int = 12, 
                         wk_col : str = 'YearWeek', wks_backwards : Union[None, int] = None, verbose : bool = False) -> list:
    
    """
    Create a list of weeks, format %Y%W, from current week to oldest week present in sales dataset.
    It is also possible to add future weeks to the list
    
    Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * current_wk : int
            * current week number in format %Y%W (e.g: 202132)
        * max_wk : int
            * maximum number of week in a year
        * len_preds : int
            * number of future weeks (at max. 26)
        * wk_col : str
            * week number column name
        * wks_backwards : int
            * number of weeks to look back, if this parameters is not None, oldest dataframe's sales week is ignored
        * verbose : bool
            * if True, some status messages are shown
        
    Returns:
    
        * wks_list(list): 
            * list of all weeks from the first dataframe's sale week to current week or future weeks, if len_preds > 0 
    """
    
    oldest_wk = df[wk_col].min()
    most_recent_wk = current_wk
    
    if verbose:
        print(oldest_wk, most_recent_wk)

    wks_list = []
    pred_window = len_preds
    wk_i = oldest_wk
    
    wk_limit = most_recent_wk + pred_window
    
    if str(wk_limit)[-2:] == str(max_wk):
        y = int(int(str(wk_limit)[-2:]) / 52)
        w = int(int(str(wk_limit)[-2:]) / y) - 51
        wk_limit = int(str( int(str(wk_limit)[:4]) + y) + str('0' + str(w))[-2:] )

    elif str(wk_limit)[-2:] > str(max_wk):
        y = int(int(str(wk_limit)[-2:]) / 52)
        w = int(int(str(wk_limit)[-2:]) / y) - 52
        wk_limit = int(str( int(str(wk_limit)[:4]) + y) + str('0' + str(w))[-2:] )
    
    while wk_i != wk_limit:
        wks_list.append(wk_i)

        if str(wk_i)[-2:] == str(max_wk):
            wk_i = int(str( int(str(wk_i)[:4]) + 1) + '01')

        else:
            wk_i += 1
            
    if wks_backwards is not None:
        strt_idx = len(wks_list[ : -len_preds]) - wks_backwards if len_preds > 0 else len(wks_list) - wks_backwards
        wks_list = wks_list[strt_idx : -len_preds] if len_preds > 0 else wks_list[strt_idx : ]

    return wks_list

def create_df_product_wk_from_wks_list(df : pd.DataFrame, wks_list : list, sku_id_col : str = 'Product', wk_col : str = 'YearWeek') -> pd.DataFrame:
    
    """
    Create a dataframe containing all combinations of a list of Products and a list of weeks
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * wks_list : list
            * list of weeks in format %Y%W (e.g: 202132)
        * sku_id_col : str
            * product column name
        * wk_col :str 
            * week number column name
    
    ### Returns:
        
        * df_all_skus_wks : pandas.DataFrame
            * dataframe containing all Product and week combinations
    """
    
    list_index = []
    
    for i in product(df[sku_id_col].unique(), np.array(wks_list)):
        list_index.append(i)

    df_all_skus_wks = pd.DataFrame(list_index, columns=[sku_id_col, wk_col])
    return df_all_skus_wks

def create_df_product_store_wk_from_wks_list(df : pd.DataFrame, wks_list : list, sku_id_col : str = 'Product', 
                                             store_id_col : str = 'Store', wk_col : str = 'YearWeek') -> pd.DataFrame:
    
    """
    Create a dataframe containing all combinations of a list of Product-Stores and a list of weeks
    
    Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * wks_list : list
            * list of weeks in format %Y%W (e.g: 202132)
        * sku_id_col : str
            * product column name
        * store_id_col : str
            * Store column name
        * wk_col : str
            * week number column name
    
    Returns:
        
        * df_all_sku_stores_wks : pandas.DataFrame
            * dataframe containing all Product-Stores and week combinations
    """
    
    list_index = []
    
    for i in product(df[sku_id_col].unique(), df[store_id_col].unique(), np.array(wks_list)):
        list_index.append(i)

    df_all_sku_stores_wks = pd.DataFrame(list_index, columns=[sku_id_col, store_id_col, wk_col])
    return df_all_sku_stores_wks

def filter_df_by_product_id(df : pd.DataFrame, sku_id : int, sku_id_col : str = 'Product', **kwargs) -> pd.DataFrame:
    
    """
    Filter a dataframe based on an Product ID
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id : int
            * product
        * sku_id_col : str
            * product column name
        
        **kwargs:
            * cols : list
                * list containing the names of all desired result dataframe's columns
    
    ### Returns:
        
        * df_filtered_by_sku : pandas.DataFrame
            * filtered dataframe
    """
    
    if kwargs['cols']:
        return df[kwargs['cols']][df[sku_id_col] == sku_id]
    else:
        return df[df[sku_id_col] == sku_id]
    
def filter_df_by_product_id_store_id(df : pd.DataFrame, sku_id : int, store_id : int, sku_id_col : str = 'Product', 
                                     store_id_col : str = 'Store', **kwargs) -> pd.DataFrame:
    
    """
    Filter a dataframe based on an Product ID and a Store ID
    
    Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id : int
            * product ID
        * store_id : int
            * store ID
        * sku_id_col : str 
            * product column name
        * store_id_col : str 
            * store column name
        
        * **kwargs:
            cols : list
                * list containing the names of all desired result dataframe's columns
    
    Returns:
        
        * df_filtered_by_sku_store : pandas.DataFrame
            * filtered dataframe
    """
    
    if kwargs['cols']:
        return df[kwargs['cols']][(df[sku_id_col] == sku_id) & (df[store_id_col] == store_id)]
    else:
        return df[(df[sku_id_col] == sku_id) & (df[store_id_col] == store_id)]

def group_df_by_week_sum(df : pd.DataFrame, wk_col : str = 'YearWeek', **kwargs) -> pd.DataFrame:
    
    """
    Groupby a dataframe by Week
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * wk_col : str
            * week number column name
    
    ### Returns:
        
        * df_grouped_by_week : pandas.DataFrame
            * original dataframe grouped by week
    """
    
    if kwargs['cols']:
        return df[kwargs['cols']].groupby(by=wk_col).sum()
    else:
        return df.groupby(by=wk_col).sum()

def availability_within_l2w(df : pd.DataFrame, sku_id_col : str = 'Product', store_id_col : str = 'Store', 
                            inv_dt_col : str = 'Date', period_days : int = 15) -> pd.DataFrame:
    
    """
    Create a dataframe containing percentage of days in shortage within last 14 days for each Product-Store
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily inventory registers
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * inv_dt_col : str
            * inventory date column name
        * period_in_days : int
            * period, in days, in which inventory will be evaluated to compute days of shortage
    
    ### Returns:
        
        * df_avail_l2w : pandas.DataFrame
            * dataframe containing percentage of days in shortage within last 14 days for each Product-Store
    """
    
    # setting cut date
    from_dt_avail = datetime.strftime(datetime.now() - timedelta(days=period_days), format='%Y-%m-%d')
    
    # filtering dataframe from cut date on
    df_avail_l2w = df[df[inv_dt_col] >= from_dt_avail]
    
    # selecting columns
    df_avail_l2w = df_avail_l2w[[sku_id_col, store_id_col]]
    
    # creating a column to count days of shortage
    df_avail_l2w['DaysOfShort'] = len(df_avail_l2w[sku_id_col]) * [1]
    
    # counting days of shortage
    df_avail_l2w = df_avail_l2w.groupby([sku_id_col, store_id_col]).sum().reset_index()
    
    # calculating share of shortage within 2 weeks
    df_avail_l2w['ShareShortageL2W'] = df_avail_l2w['DaysOfShort'].apply(lambda x: round(x / 14, 2))
    
    # returning final dataframe
    return df_avail_l2w

def calculate_safety_stock(df : pd.DataFrame, df_oct : pd.DataFrame, df_service_level : pd.DataFrame, sku_id_col : str = 'Product', 
                           store_id_col : str = 'Store', sls_qty_col : str = 'SaleQty', dt_col : str = 'Date', z_col : str = 'Z', 
                           period_days : int = 365, adjust_outliers : bool = True, threshold : float = 2) -> pd.DataFrame:
    
    """
    Compute safety stock quantity for each Product-Store
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily inventory registers
        * df_oct : pandas.DataFrame
            * dataframe containing OCT (order cycle time) of each Product-Store
        * df_service_level : pandas.DataFrame
            * dataframe containing service level rate for each ABC x PQR combination (index must have [A, B, C] values and columns [X, Y, Z] values)
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * sls_qty_col : str
            * sales quantity column name
        * dt_col : str
            * sale date (%Y-%m-%d) column name
        * z_col : str
            * z-value (safety stock's formula coefficient) column name
        * period_in_days : int
            * period, in days, in which sales will be evaluated to compute safety stock
        * adjust_outliers : bool
            * if True, outlier are adjusted to average through IQR method
        * threshold : float
            * multiplicative factor used to set lower and upper outlier bounds
    
    ### Returns:
        
        * df_ss : pandas.DataFrame
            * dataframe containing the safety stock for each Product-Store
    """
    
    # setting cut date
    from_dt_ss = datetime.strftime(datetime.now() - timedelta(days=period_days), format='%Y-%m-%d')
    
    if adjust_outliers:

        # grouping by df_sls_l730d and calculating q1 and q3
        df_iqr = pd.merge(df[[sku_id_col, store_id_col, sls_qty_col]].groupby(by=[sku_id_col, store_id_col]).quantile(0.25).reset_index().rename(columns={sls_qty_col: 'Q1'}), 
                           df[[sku_id_col, store_id_col, sls_qty_col]].groupby(by=[sku_id_col, store_id_col]).quantile(0.75).reset_index().rename(columns={sls_qty_col: 'Q3'}), 
                           on=[sku_id_col, store_id_col], how='left')

        # calculating iqr
        df_iqr['IQR'] = df_iqr['Q3'] - df_iqr['Q1']

        # grouping by df_sls_l730d and calculating average
        df_iqr = pd.merge(df_iqr, 
                           df[[sku_id_col, store_id_col, sls_qty_col]].groupby(by=[sku_id_col, store_id_col]).mean().reset_index().rename(columns={sls_qty_col: 'Avg(SaleQty)'}), 
                           on=[sku_id_col, store_id_col], how='left')

        # removing negative values and filling na with 0
        df_iqr[df_iqr['Avg(SaleQty)'] < 0] = 0
        df_iqr['Avg(SaleQty)'].fillna(0, inplace=True)

        # grouping by df_sls_l730d and calculating median
        df_iqr = pd.merge(df_iqr, 
                           df[[sku_id_col, store_id_col, sls_qty_col]].groupby(by=[sku_id_col, store_id_col]).median().reset_index().rename(columns={sls_qty_col: 'Median(SaleQty)'}), 
                           on=[sku_id_col, store_id_col], how='left')

        # removing negative values and filling na with 0
        df_iqr[df_iqr['Median(SaleQty)'] < 0] = 0
        df_iqr['Median(SaleQty)'].fillna(0, inplace=True)

        # setting lower and upper bounds to remove outliers
        df_iqr['LowerIQR'] = df_iqr.apply(lambda x: x['Q1'] - threshold * x['IQR'], axis=1)
        df_iqr['UpperIQR'] = df_iqr.apply(lambda x: x['Q3'] + threshold * x['IQR'], axis=1)


        # merging iqr infos to sales dataframe
        df = pd.merge(df[df[dt_col] >= from_dt_ss][[sku_id_col, store_id_col, sls_qty_col]], 
                      df_iqr, on=[sku_id_col, store_id_col], how='left')

        # filling possible na values with 0
        df[['Q1', 'Q3', 'IQR', 'Avg(SaleQty)', 'Median(SaleQty)', 'LowerIQR', 'UpperIQR']].fillna(0, inplace=True)

        # classifying outliers using iqr method ( (x < q1 - threshold * iqr)   or  ( x > q3 + threshold * iqr) )
        df['Outlier'] = df.apply(lambda x: 1 if (x[sls_qty_col] < x['LowerIQR']) or (x[sls_qty_col] > x['UpperIQR']) else 0, axis=1)

        # adjusting outliers with average
        df['SaleQtyAdj'] = df.apply(lambda x: x[sls_qty_col] if x['Outlier'] == 0 else x['Avg(SaleQty)'], axis=1)

        # dropping original sales column
        df.drop(columns=sls_qty_col, inplace=True)

        # renaming adjusted sales column
        df.rename(columns={'SaleQtyAdj': sls_qty_col}, inplace=True)
        
    else:
        
        #filtering just sales registers inside the desired period
        df = df[df[dt_col] >= from_dt_ss]

    # working with days instead of weeks, otherwise it would be needed to convert lead time from days to weeks
    df_ss = df[[sku_id_col, store_id_col, sls_qty_col]].groupby([sku_id_col, store_id_col]).std().fillna(0).rename(columns={sls_qty_col: 'WkSaleStd'}).reset_index()

    # getting Z value for each product
    df_ss = pd.merge(df_ss, df_service_level[[sku_id_col, z_col]], on=sku_id_col, how='left')
    
    # getting lead time (oct) for each store
    df_ss = pd.merge(df_ss, df_oct, on=store_id_col, how='left').fillna(0)
    
    # calculating safety stock
    df_ss['Safety stock'] = df_ss[['WkSaleStd', 'OCT', 'Z']].apply(lambda x: x['Z'] * x['WkSaleStd'] * np.sqrt(x['OCT']), axis=1)

    # returning final dataframe
    return df_ss.sort_values('Safety stock', ascending=False)

def calc_weekly_sales_share(df : pd.DataFrame, sku_id_col : str = 'Product', store_id_col : str = 'Store', wk_col : str = 'YearWeek', 
                            sls_qty_col : str = 'SaleQty') -> pd.DataFrame:
    
    """
    Compute weekly sales share for each Product-Store
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * wk_col : str
            * week number column name
        * sls_qty_col : str
            * sales quantity column name
    
    ### Returns:
        
        * df_wk_share : pandas.DataFrame
            * dataframe containing weekly sales share for each Product-Store
    """
    
    # merging dataframes
    df_wk_share = pd.merge(
        df[[wk_col, sku_id_col, store_id_col, sls_qty_col]].groupby([wk_col, sku_id_col, store_id_col]).sum().reset_index(), 
        df[[wk_col, sku_id_col, sls_qty_col]].groupby([wk_col, sku_id_col]).sum().reset_index(), 
        on=[wk_col, sku_id_col], how='outer', suffixes=('_Wk_Product_Store', '_Wk_Product')
    )
    
    # calculating sale share
    df_wk_share['Share'] = round(df_wk_share[f'{sls_qty_col}_Wk_Product_Store'] / df_wk_share[f'{sls_qty_col}_Wk_Product'], 4)

    # returning final dataframe
    return df_wk_share

def calc_sales_share_l8w(df : pd.DataFrame, wks_list : list, df_eleg : Union[None, pd.DataFrame] = None, wks_backwards : int = 8, 
                         sku_id_col : str = 'Product', store_id_col : str = 'Store', wk_col : str = 'YearWeek', 
                         sls_qty_col : str = 'SaleQty') -> pd.DataFrame:
    
    """
    Compute sales share for each Product-Store within a range of last weeks
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily sales registers
        * wks_list : list
            * list of weeks in format %Y%W (e.g: 202132)
        * df_eleg : pandas.DataFrame
            * dataframe containing only elegible Product-Store combinations (one column for Product ID and another for Store ID)
        * wks_backwards : int
            * number of weeks to look back, must be greater than or equal to 1
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * wk_col : str
            * week number column name
        * sls_qty_col : str
            * sales quantity column name
    
    ### Returns:
        
        * df_share_lxw : pandas.DataFrame
            * dataframe containing sales share for each Product-Store within last weeks
    """
    
    wks_list = wks_list[-wks_backwards:]
    df = df[df[wk_col].isin(wks_list)]
    df[sls_qty_col] = df[sls_qty_col].clip(0)
    
    
    if df_eleg is not None:
        # creating list of tuples of sku-store combinations
        eleg_tuples = df_eleg[[sku_id_col, store_id_col]].apply(tuple, axis=1).tolist()
        # removing duplicated tuples if there is any
        eleg_tuples = list(set([i for i in eleg_tuples]))
        # filtering original dataframe to remove not-elegible combinations sku-store
        df = df[(df[[sku_id_col, store_id_col]].apply(tuple, axis=1).isin(eleg_tuples))]
    
    # merging dataframes
    df_share_l8w = pd.merge(
        df[[sku_id_col, store_id_col, sls_qty_col]].groupby([sku_id_col, store_id_col]).sum().reset_index(), 
        df[[sku_id_col, sls_qty_col]].groupby(sku_id_col).sum().reset_index(), 
        on=sku_id_col, how='outer', suffixes=('_Wk_Product_Store', '_Wk_Product')
)   
   # calculating sale share
    df_share_l8w['ShareL8W'] = round(df_share_l8w[f'{sls_qty_col}_Wk_Product_Store'] / df_share_l8w[f'{sls_qty_col}_Wk_Product'], 4)
    
    # filling possible nan values with 0 (result of 0 divided by 0)
    df_share_l8w['ShareL8W'].fillna(0, inplace=True)

    # returning final dataframe
    return df_share_l8w

def calculate_sell_through(df : pd.DataFrame, df_inv_store : pd.DataFrame, period_in_days : int = 7, sku_id_col : str = 'Product', 
                           store_id_col : str = 'Store', sls_qty_col : str = 'SaleQty', sku_stk_col : str = 'StockQty', 
                           dt_col : str = 'Date') -> pd.DataFrame:
    
    """
    Compute sell through indicator for each Product-Store
    Sell Through is percentage of inventory sold within last days (default: 7 days)
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily inventory registers
        * df_inv_store : pandas.DataFrame
            * dataframe containing yesterday's inventory of each Product-Store
        * period_in_days : int
            * period, in days, in which sales will be considered to compute sell through
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * sls_qty_col : str
            * sales quantity column name
        * sku_stk_col : str
            * store inventory column name
        * dt_col : str
            * sale date (%Y-%m-%d) column name
    
    ### Returns:
        
        * df_sell_through : pandas.DataFrame
            * dataframe containing sell through indicator for each Product-Store
    """
    
    period_in_days += 1
    
    # setting cut date
    from_dt_st = datetime.strftime(datetime.now() - timedelta(days=period_in_days + 1), format='%Y-%m-%d')
    # filtering sales dataframe from cut date on
    sls_sku_store_l7d = df[df[dt_col] >= from_dt_st][[sku_id_col, store_id_col, sls_qty_col]].groupby([sku_id_col, store_id_col]).sum().reset_index()
    # replacing negative values for 0
    sls_sku_store_l7d_sls = sls_sku_store_l7d[sls_qty_col]
    sls_sku_store_l7d_sls = np.where(sls_sku_store_l7d_sls < 0, 0, sls_sku_store_l7d_sls)
    sls_sku_store_l7d[sls_qty_col] = sls_sku_store_l7d_sls
    # getting all sku-store yesterday's on hand inventory
    df_yest_inv_stores = df_inv_store[[sku_id_col, store_id_col, sku_stk_col]]
    # merging boht dataframes
    df_sell_through = pd.merge(sls_sku_store_l7d, df_yest_inv_stores, on=[sku_id_col, store_id_col], how='outer').fillna(0)
    # calculating sell through
    df_sell_through['SellThrough'] = df_sell_through.apply(lambda x: round(x[sls_qty_col] / ( x[sls_qty_col] + x[sku_stk_col] ), 2) if ( x[sls_qty_col] + x[sku_stk_col] ) != 0 else 0, axis=1)
    # filling nan with 0 because they had no sales
    df_sell_through['SellThrough'].fillna(0, inplace=True)
    # returning final dataframe
    return df_sell_through

def calculate_inventory_slope(df_inv : pd.DataFrame, period_in_days : int = 7, sku_id_col : str = 'Product', store_id_col : str = 'Store', 
                              sku_stk_col : str = 'StockQty', dt_col : str = 'Date', use_df_last_day : bool = False) -> pd.DataFrame:
    
    """
    Compute inventory slope within a range of last days
    This slope indicates how fast inventory has decreased or incresead within last days
    
    ### Parameters:
    
        * df_inv_store : pandas.DataFrame
            * dataframe containing yesterday's inventory of each Product-Store
        * period_in_days : int
            * period, in days, in which inventory will be considered to compute inventory slope
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * sku_stk_col : str
            * store inventory column name
        * dt_col : str
            * sale date (%Y-%m-%d) column name
        * use_df_last_day : bool
            * if True, consider most recent date in df_inv and analysis the past from there, if False, consider yesterday's date
    
    ### Returns:
        
        * df_slopes : pandas.DataFrame
            * dataframe containing inventory slope indicator for each Product-Store
    """
    
    if use_df_last_day:
        list_l7d = [datetime.strftime(df_inv[dt_col].max() - timedelta(days=d), format='%Y-%m-%d') for d in range(period_in_days,0,-1)]
    else:
        list_l7d = [datetime.strftime(datetime.now() - timedelta(days=d), format='%Y-%m-%d') for d in range(period_in_days,0,-1)]

    df_inv_sku_dt = pd.DataFrame(list(
        product(list(df_inv[sku_id_col].unique()), 
                list(df_inv[store_id_col].unique()), 
                list_l7d)), 
        columns=[sku_id_col, store_id_col, dt_col]
    )
    
    df_inv = df_inv.copy()
    
    df_inv[dt_col] = pd.to_datetime(df_inv[dt_col], format="%Y-%m-%d")
    df_inv_sku_dt[dt_col] = pd.to_datetime(df_inv_sku_dt[dt_col], format="%Y-%m-%d")
    
    df_inv_l7d = pd.merge(df_inv_sku_dt, 
                          df_inv[[sku_id_col, store_id_col, dt_col, sku_stk_col]], 
                          on=[sku_id_col, store_id_col, dt_col], how='left')

    df_inv_l7d.fillna(0, inplace=True)

    sku_store_tuple = df_inv_l7d[[sku_id_col, store_id_col]].apply(tuple, axis=1)

    sku_store_tuple = list(set([t for t in sku_store_tuple]))

    x_axis = [x for x in range(len(list_l7d))]

    dict_slopes = dict()

    for sku, store in tqdm(sku_store_tuple):
        inv_hist = df_inv_l7d[(df_inv_l7d[sku_id_col] == sku) & (df_inv_l7d[store_id_col] == store)].sort_values(dt_col, ascending=True)[sku_stk_col].tolist()
        linresult = linregress([x for x in range(7)], inv_hist)
        dict_slopes[(sku, store)] = linresult.slope

    df_slopes = pd.DataFrame(data=dict_slopes.values(), index=dict_slopes.keys())
    df_slopes.index.names = [sku_id_col, store_id_col]
    df_slopes.columns = ['Slope']
    return df_slopes.reset_index()

def create_df_product_store_sls_l8w(df : pd.DataFrame, df_eleg : pd.DataFrame, current_wk : int, wks_backwards : int = 12, 
                                    sku_id_col : str = 'Product', store_id_col : str = 'Store', wk_col : str = 'YearWeek', 
                                    sls_qty_col : str = 'SaleQty') -> pd.DataFrame:
    
    """
    Create a dataframe containing weekly sales quantity (one column per week) for each Product-Store
    This method assures that all elegible Product-Stores are considered
    
    ### Parameters:
    
        * df : pandas.DataFrame
            * dataframe containing all daily inventory registers
        * df_eleg : pandas.DataFrame
            * dataframe containing only elegible Product-Store combinations (one column for Product ID and another for Store ID)
        * current_wk : int
            * current week number in format %Y%W (e.g: 202132)
        * wks_backwards : int
            * number of weeks to look back, must be greater than or equal to 1
        * len_preds : int
            * number of future weeks
        * sku_id_col : str
            * product ID column name
        * store_id_col : str
            * store ID column name
        * wk_col : str
            * week number column name
        * sls_qty_col : str
            * sales quantity column name
    
    ### Returns:
        
        * df_sls_lw : pandas.DataFrame
            * dataframe containing weekly sales quantity (one column per week) for each Product-Store
    """
    
    # creating list of weeks and rolling back {wks_backwards} periods
    l8w_list = create_list_of_weeks(df, current_wk, wks_backwards=12, len_preds=0)
    # filtering dataframe by list of weeks and grouping by sku_id_col, store_id_col and wk_col
    df_sls_lw = df[df[wk_col].isin(l8w_list)][[sku_id_col, store_id_col, wk_col, sls_qty_col]].groupby([sku_id_col, store_id_col, wk_col]).sum()
    # unstacking wk_col to have 1 column for each week, also filling nan values with 0
    df_sls_lw = df_sls_lw.unstack().fillna(0)
    # renaming columns to have only 1 level
    renamed_cols = [wk for (i, wk) in df_sls_lw.columns]
    df_sls_lw.columns = renamed_cols
    # creating a new column with the sum of last {wks_backwards} weeks
    df_sls_lw[f'SaleQtyL{wks_backwards}W'] = df_sls_lw.apply(sum, axis=1)
    # resetting index
    df_sls_lw.reset_index()
    # merging df_eleg to df_sls_lw to have all elegible sku_stores considered
    df_sls_lw = pd.merge(df_eleg[[sku_id_col, store_id_col]], df_sls_lw, on=[sku_id_col, store_id_col], how='left')
    # filling nan values with 0 in all sales columns (weekly and total)
    df_sls_lw[renamed_cols + [f'SaleQtyL{wks_backwards}W']] = df_sls_lw[renamed_cols + [f'SaleQtyL{wks_backwards}W']].fillna(0)
    return df_sls_lw

def calculate_forecast_oct_wks(x) -> float:
    
    """
    Compute forecast quantity for order cycle time (OCT) period
    
    Applied on pandas.DataFrame's columns ['OCT_WKS', 'W+0', 'W+1', ... , 'W+{i}']
    """
    
    oct_wks_int = int(x['OCT_WKS'])
    oct_wks_residual = x['OCT_WKS'] - oct_wks_int
    
    forecast_oct = sum(x[[f'W+{i}' for i in range(oct_wks_int)]])
    
    forecast_oct += (x[[f'W+{i}' for i in range( oct_wks_int + 1 )][-1]]) * oct_wks_residual
    
    return forecast_oct

def calculate_forecast_oct_wks_from_projection(x, len_preds : int = 12) -> float:
    
    """
    Compute forecast quantity for second order cycle time (OCT) period (1 OCT ahead from now)
    
    Applied on pandas.DataFrame's columns ['OCT_WKS', 'W+0', 'W+1', ... , 'W+{i}']
    """
    
    oct_wks_ceil = np.ceil(x['OCT_WKS'])
    oct_wks_1st_residual = oct_wks_ceil - x['OCT_WKS']
    oct_left = x['OCT_WKS'] - oct_wks_1st_residual
    oct_wks_int = int(oct_left)
    oct_wks_2nd_residual = oct_left - oct_wks_int
    
    # getting just weekly forecast from projection
    useful_wk_forecast = x[[f'W+{i}' for i in range( max(0, int(oct_wks_ceil) - 1 ), len_preds)]]
    
    # first residual forecast
    resid_1st_forecast = useful_wk_forecast[0] * (oct_wks_1st_residual)
    
    # forecast of next complete weeks
    complete_wks_forecast = sum([frcst for frcst in useful_wk_forecast[1 : oct_wks_int+1]])
    
    # final residual forecast
    resid_2nd_forecast = useful_wk_forecast[oct_wks_int + 1] * oct_wks_2nd_residual
    
    forecast_oct_from_projection = resid_1st_forecast + complete_wks_forecast + resid_2nd_forecast

    return forecast_oct_from_projection

def calculate_coverage_ss(x) -> float:
    
    """
    Compute safety stock coverage considering weekly forecast
    
    Applied on pandas.DataFrame's columns ['Safety stock', 'W+0', 'W+1', ... , 'W+{i}']
    """
    
    ss_value = x['Safety stock']
    
    array_cumsum = np.cumsum(x[[f'W+{i}' for i in range(12)]])
    array_preds = x[[f'W+{i}' for i in range(12)]]

    bigger_than_ss = (array_cumsum > ss_value).sum()
    int_cover = len(array_cumsum) - bigger_than_ss

    if int_cover >= len(array_cumsum):
        
        return '12+'
    
    elif int_cover == 0:
        
        residual_cover = ss_value / array_cumsum[int_cover - 1]
        
    else:
        
        residual_cover = (ss_value - array_cumsum[int_cover - 1]) / array_preds[int_cover]
        
    coverage_ss = int_cover + residual_cover
    
    return coverage_ss

def calculate_avg_coverage_ss(x) -> float:
    
    """
    Compute safety stock coverage considering average forecast N12W
    
    Applied on pandas.DataFrame's columns ['Safety stock', 'AvgForecastQtN12W']
    """
    
    if (x['AvgForecastQtN12W'] <= 0) and (x['Safety stock'] <= 0):
        
        avg_cover_ss = 0
        
    elif (x['AvgForecastQtN12W'] <= 0) and (x['Safety stock'] > 0):
    
        avg_cover_ss = 9999
        
    else:
    
        avg_cover_ss = round(x['Safety stock'] / x['AvgForecastQtN12W'], 2)
    
    return avg_cover_ss

def calculate_coverage_stk(x) -> float:
    
    """
    Compute inventory coverage considering weekly forecast
    
    Applied on pandas.DataFrame's columns ['TotalStockQty', 'W+0', 'W+1', ... , 'W+{i}']
    """
    
    stk_value = x['TotalStockQty']
    forecast_sum = sum(x[[f'W+{i}' for i in range(12)]])
    
    if (forecast_sum <= 0) and (stk_value <= 0) :
        
        coverage_stk = 0
        
    if (forecast_sum <= 0) and (stk_value > 0):
    
        coverage_stk = 9999
    
    else:

        array_cumsum = np.cumsum(x[[f'W+{i}' for i in range(12)]])
        array_preds = x[[f'W+{i}' for i in range(12)]]

        bigger_than_ss = (array_cumsum > stk_value).sum()
        int_cover = len(array_cumsum) - bigger_than_ss

        if int_cover >= len(array_cumsum):

            return '12+'

        elif int_cover == 0:

            residual_cover = stk_value / array_cumsum[int_cover - 1]

        else:

            residual_cover = (stk_value - array_cumsum[int_cover - 1]) / array_preds[int_cover]

        coverage_stk = int_cover + residual_cover
    
    return coverage_stk

def calculate_avg_coverage_stk(x) -> float:
    
    """
    Compute safety stock coverage considering average forecast N12W
    
    Applied on pandas.DataFrame's columns ['TotalStockQty', 'AvgForecastQtN12W']
    """
    
    if (x['AvgForecastQtN12W'] <= 0) and (x['TotalStockQty'] <= 0):
        
        avg_cover_stk = 0
        
    elif (x['AvgForecastQtN12W'] <= 0) and (x['TotalStockQty'] > 0):
    
        avg_cover_stk = 9999
        
    else:
    
        avg_cover_stk = round(x['TotalStockQty'] / x['AvgForecastQtN12W'], 2)
    
    return avg_cover_stk

def calculate_avg_coverage_proj_inv(x) -> float:
    
    """
    Compute safety stock coverage considering average forecast N12W
    
    Applied on pandas.DataFrame's columns ['ProjInv_1OCT_ahead', 'AvgForecastQtN12W']
    """
    
    if (x['AvgForecastQtN12W'] <= 0) and (x['ProjInv_1OCT_ahead'] <= 0) :
        
        avg_cover_proj_inv = 0
        
    elif (x['AvgForecastQtN12W'] <= 0) and (x['ProjInv_1OCT_ahead'] > 0):
    
        avg_cover_proj_inv = 9999
        
    else:
    
        avg_cover_proj_inv = round(x['ProjInv_1OCT_ahead'] / x['AvgForecastQtN12W'], 2)
    
    return avg_cover_proj_inv

def projected_inv_one_oct_ahead(x) -> float:
    
    """
    Compute projected inventory one OCT ahead considering current inventory and forecast during the period
    
    Applied on pandas.DataFrame's columns ['TotalStockQty', 'Forecast_OCT_WKS']
    """
    
    proj_inv = x['TotalStockQty'] - x['Forecast_OCT_WKS']
    
    proj_inv = np.round(np.where(proj_inv < 0, 0, proj_inv), 2)
    
    return proj_inv

def filter_yesterday_inv_store(df_inv : pd.DataFrame, dt_col : str = 'Date', inv_qty_col : str = 'StockQty', 
                               inv_trns_qty_col : str = 'StockQtyInTransit', sum_oh_intrn : bool = True, 
                               sum_oh_intrn_col : str = 'TotalStockQty', base_zero : bool = True, 
                               cut_dt : Union[None, str] = None, use_df_last_day : bool = False) -> pd.DataFrame:
    
    """
    Filter inventory dataframe to get only yesterday's store inventory registers 
    
    ### Parameters:
    
        * df_inv : pandas.DataFrame
            * dataframe containing inventory registers of each Product-Store
        * dt_col : str
            * stock date column name
        * inv_qty_col : str
            * stock quantity column name
        * inv_trns_qty_col : str
            * stock in transit quantity column name
        * sum_oh_intrn : bool
            * if True, sum up On Hand + In Transit Inventories, otherwise On Hand Inventory will be used
        * sum_oh_intrn_col : str
            * Oh Hand + In Transit Inventory column name
        * base_zero : bool
            * if True, correct negative register to zero
        * cut_dt : str
            * desired inventory date formatted as "%Y-%m-%d" (ex: '2021-03-25'), if it's set as None, 'use_df_last_day' parameter will be checked
        * use_df_last_day : bool
            * if True, the dataframe most recent date will be used, otherwise yesterday date will be used
        
    ### Returns:
    
        * df_inv_store_l1d : pandas.DataFrame
            * dataframe containing only yesterday's inventory values for each Product-Store
    """
    
    if cut_dt is None:
    
        if use_df_last_day:
            cut_dt = df_inv[dt_col].max() - timedelta(days=1)
        else:
            cut_dt = datetime.now() - timedelta(days=1)
            
    if base_zero:
        inv_series = df_inv[inv_qty_col]
        df_inv[inv_qty_col] = np.where(inv_series < 0, 0, inv_series)
        
    if sum_oh_intrn:
        df_inv[sum_oh_intrn_col] = df_inv[inv_qty_col] + df_inv[inv_trns_qty_col]
            
    return df_inv[(df_inv[dt_col] == cut_dt)]

def filter_yesterday_inv_dc(df_inv : pd.DataFrame, specify_dc : list, store_id_col : str = 'Store', 
                            dt_col : str = 'Date', inv_qty_col : str = 'StockQty', base_zero : bool = True, 
                            cut_dt : Union[None, str] = None, use_df_last_day : bool = False) -> pd.DataFrame:
    
    """
    Filter inventory dataframe to get only yesterday's DC inventory registers 
    
    ### Parameters:
    
        * df_inv : pandas.DataFrame
            * dataframe containing inventory registers of each Product-DC
        * specify_cd : list
            * must receive a list with all desired DC IDs
        * store_id_col : str
            * DC ID column name
        * dt_col : str
            * stock date column name
        * inv_qty_col : str
            * stock quantity column name
        * base_zero : bool
            * if True, correct negative register to zero
        * cut_dt : str
            * desired inventory date formatted as "%Y-%m-%d" (ex: '2021-03-25'), if it's set as None, 'use_df_last_day' parameter will be checked
        * use_df_last_day : bool
            * if True, the dataframe most recent date will be used, otherwise yesterday date will be used
        
    ### Returns:
    
        * df_inv_cd_l1d : pandas.DataFrame
            * dataframe containing only yesterday's inventory values for each Product-DC
    """
    
    if cut_dt is None:
    
        if use_df_last_day:
            cut_dt = df_inv[dt_col].max() - timedelta(days=1)
        else:
            cut_dt = datetime.now() - timedelta(days=1)
            
    if base_zero:
        inv_series = df_inv[inv_qty_col]
        df_inv[inv_qty_col] = np.where(inv_series < 0, 0, inv_series)

    return df_inv[(df_inv[dt_col] == cut_dt) & (df_inv[store_id_col].isin(specify_dc))]

def filter_lxd_inv_store(df_inv : pd.DataFrame, period_in_days: int, store_id_col : str = 'Store', dt_col : str = 'Date', 
                         inv_qty_col : str = 'StockQty', base_zero : bool = True, cut_dt : Union[None, str] = None, 
                         use_df_last_day: bool  = False) -> pd.DataFrame:
    
    """
    Filter inventory dataframe to get store inventory registers within a range of last days
    
    ### Parameters:
    
        * df_inv : pandas.DataFrame
            * dataframe containing inventory registers of each Product-Store
        * period_in_days : int
            * period, in days, in which dates will be considered to filter original dataframe
        * store_id_col : str
            * store ID column name
        * dt_col : str
            * inventory date column name
        * inv_qty_col : str
            * inventory quantity column name
        * base_zero : bool
            * if True, correct negative register to zero
        * cut_dt : str
            * desired inventory date formatted as "%Y-%m-%d" (ex: '2021-03-25'), if it's set as None, 'use_df_last_day' parameter will be checked
        * use_df_last_day : bool
            * if True, the dataframe most recent date will be used, otherwise yesterday date will be used
        
    ### Returns:
    
        * df_inv_cd_lxd : pandas.DataFrame
            * dataframe containing last days' inventory values for each Product-Store
    """
    
    if cut_dt is None:
    
        if use_df_last_day:
            cut_dt = df_inv[dt_col].max() - timedelta(days=period_in_days+1)
        else:
            cut_dt = datetime.now() - timedelta(days=period_in_days+1)
            
    if base_zero:
        inv_series = df_inv[inv_qty_col]
        df_inv[inv_qty_col] = np.where(inv_series < 0, 0, inv_series)
            
    return df_inv[(df_inv[dt_col] >= cut_dt)]