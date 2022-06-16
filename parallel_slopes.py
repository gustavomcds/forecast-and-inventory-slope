import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from multiprocessing import Pool
from functools import partial

from distribution.auxiliary import filter_lxd_inv_store, calculate_inventory_slope

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

    filepath = os.path.join(os.getcwd(), 'database')
    filename = 'STOCK_DLY.csv'

    print("Reading inventory data")

    df_inventory = pd.read_csv(os.path.join(filepath, filename))

    df_inventory.drop(df_inventory.columns[0], axis=1)
    df_inventory['Date'] = pd.to_datetime(df_inventory['Date'], utc=False)
    df_inventory['YearWeek'] = df_inventory['Date'].dt.strftime('%Y%W').astype(int)

    print("Inventory data is ready")

    df_inv_store_l7d = filter_lxd_inv_store(df_inventory, period_in_days=7, use_df_last_day=True)

    print(df_inv_store_l7d)

    n_cores = 4
    unique_items_parallel_all = df_inv_store_l7d['Product'].unique()
    unique_items_parallel_split = np.array_split(unique_items_parallel_all, n_cores)

    list_of_dfs_split = [df_inv_store_l7d[df_inv_store_l7d['Product'].isin(list_of_items_split)] for list_of_items_split in unique_items_parallel_split]

    calculate_inventory_slope_partial = partial(calculate_inventory_slope, use_df_last_day=True)

    print('Starting slope calculations')
    
    start_time = time.time()
    
    with Pool(n_cores) as pool:

        df_pool = pd.concat(pool.map(calculate_inventory_slope_partial, list_of_dfs_split))
        
    print(f"Execution time: {time.time() - start_time}")
        
    today_str = datetime.strftime(datetime.now(), format='%Y%m%d')
    df_pool.to_excel(os.path.join(os.getcwd(), 'results', f'{today_str}_df_slopes.xlsx'), index=False)