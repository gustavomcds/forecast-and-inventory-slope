import pandas as pd
import numpy as np

from distribution.auxiliary import filter_df_by_product_id, create_list_of_weeks, group_df_by_week_sum, filter_df_by_product_id_store_id

def create_series_one_product_ready_to_prediction(df : pd.DataFrame, product_id : str, current_wk, product_id_col : str ='Product', 
                                                  wk_col : str ='YearWeek', sls_qty_col : str ='SaleQty', verbose : bool =False) -> pd.Series:
    
    """
    Create a time series of one product, sorted by time, from oldest to most recent
    Dataframe is filtered by Product ID, grouped and sorted by week.
    Weeks with no sales are filled with 0
    
    ### Parameters:
    * df : pandas.DataFrame
        * pandas DataFrame containing daily sales registers
    * product_id : str 
        * product ID
    * current_wk : int 
        * current year-week number formatted as %Y%W (e.g: 201901)
    * product_id_col : str 
        * product ID column name
    * wk_col : str
        * week number column name
    * sls_qty_col : str
        * sales quantity column name
    * verbose : bool
        * if True, some status messages are shown
    
    ### Returns:
    * series_one_product : pd.Series
        * pandas Series containing one product's historical sales
    """
    
    df = filter_df_by_product_id(df, product_id, cols=[wk_col, product_id_col, sls_qty_col])
    wks_list = create_list_of_weeks(df, current_wk, len_preds=0, verbose=verbose)
    df = group_df_by_week_sum(df, cols=[wk_col, sls_qty_col])
    df.sort_values(by=wk_col, inplace=True)
    start_wk = df.index.min()
    start_idx = wks_list.index(start_wk)
    
    wks_list = wks_list[ start_idx : ]
    
    df_all_wks = pd.DataFrame(wks_list, columns=[wk_col])
    
    series_one_product = pd.merge(df_all_wks, df, on=[wk_col], how='left').set_index(wk_col).fillna(0)[sls_qty_col]
    
    return series_one_product

def create_series_product_store_ready_to_prediction(df : pd.DataFrame, product_id : str, store_id : str, current_wk : int, product_id_col : str ='Product', 
                                                    store_id_col : str = 'Store', wk_col : str = 'YearWeek', sls_qty_col : str = 'SaleQty', verbose : bool =False) -> pd.Series:
    
    """
    Create a time series of one Product-Store, sorted by time, from oldest to most recent.
    Dataframe is filtered by Product ID and Store ID, grouped and sorted by week.
    Weeks with no sales are filled with 0.
    
    ### Parameters:
    * df : pandas.DataFrame
        * dataframe containing daily sales registers
    * product_id : str
        * product ID
    * store_id : str 
        * store ID
    * current_wk : int
        * current year-week number formatted as %Y%W (e.g: 201901)
    * product_id_col : str
        * product ID column name
    * store_id_col : str 
        * store ID column name
    * wk_col : str
        * week number column name
    * sls_qty_col : str
        * sales quantity column name
    * verbose : bool 
        * if True, some status messages are shown
    
    ### Returns:
    * series_one_product_store : pandas.Series: 
        * series containing one product-store's historical sales
    """
    
    df = filter_df_by_product_id_store_id(df, product_id, store_id, cols=[wk_col, product_id_col, store_id_col, sls_qty_col])
    wks_list = create_list_of_weeks(df, current_wk, len_preds=0, verbose=verbose)
    df = group_df_by_week_sum(df, cols=[wk_col, sls_qty_col])
    df.sort_values(by=wk_col, inplace=True)
    start_wk = df.index.min()
    start_idx = wks_list.index(start_wk)
    
    wks_list = wks_list[ start_idx : ]
    
    df_all_wks = pd.DataFrame(wks_list, columns=[wk_col])
    
    series_one_product_store = pd.merge(df_all_wks, df, on=[wk_col], how='left').set_index(wk_col).fillna(0)[sls_qty_col]
    
    return series_one_product_store