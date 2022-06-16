import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt

from distribution.auxiliary import create_list_of_weeks
from forecast.ensemble import predict_many_skus_based_on_dataframe

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    filepath = os.path.join(os.getcwd(), 'database')
    filename = 'SALES_DLY.csv'

    print("Reading sales data")

    df_sls = pd.read_csv(os.path.join(filepath, filename))

    df_sls.drop(df_sls.columns[0], axis=1)
    df_sls['Date'] = pd.to_datetime(df_sls['Date'], utc=False)
    df_sls['YearWeek'] = df_sls['Date'].dt.strftime('%Y%W').astype(int)

    print("Sales data is ready")

    current_wk = df_sls['YearWeek'].max()
    print(f"Current week: {current_wk}")

    wks_list_filter_sls = create_list_of_weeks(df_sls, current_wk, len_preds=0)
    df_sls_filtered = df_sls[df_sls['YearWeek'] <= wks_list_filter_sls[-1]]

    n_cores = 4
    unique_items_parallel_all = df_sls_filtered['Product'].unique()
    unique_items_parallel_split = np.array_split(unique_items_parallel_all, n_cores)

    list_of_dfs_split = [df_sls_filtered[df_sls_filtered['Product'].isin(list_of_items_split)] for list_of_items_split in unique_items_parallel_split]
      
    predict_many_items_partial = partial(predict_many_skus_based_on_dataframe,  
                                         current_wk=current_wk)
    
    start_time = time.time()
    
    with Pool(n_cores) as pool:

        df_pool = pd.concat(pool.map(predict_many_items_partial, list_of_dfs_split))
        
    print(f"Execution time: {time.time() - start_time}")
        
    today_str = datetime.strftime(datetime.now(), format='%Y%m%d')
    df_pool.to_excel(os.path.join(os.getcwd(), 'results', f'{today_str}_forecast_results.xlsx'), index=False)