# Forecast and inventory slope

## What is this project about?

This project has two functionalities: 

- **calculate forecast** using Triple Exponential Smoothing and Moving Average (choose the best of them based on an error metric)
- **compute inventory linear regression's slope** to determine the speed of inventory increase/decrease and its magnitude.

## Setup

**Cloning the repository**
```
$ git clone git@github.com:gustavomcds/forecast-and-inventory-slope.git
$ cd forecast-and-inventory-slope
$ python parallel_forecast.py
$ python parallel_slopes.py
```

**Installing dependencies**
```
$ pip install -r requirements.txt
```

## How to use

Currently, the code supports just csv files as input.

Two files are needed which must be stored in <a href="https://github.com/gustavomcds/forecast-and-inventory-slope/tree/main/database">database</a> folder:

- SALES_DLY.csv: contains the daily sales history for each Product (the bigger the data period, the better)
- STOCK_DLY.csv: contains the daily inventory quantity history for each Product-Store within, at least, the last 7 days

## Outputs

There are two outputs which are saved in <a href="https://github.com/gustavomcds/forecast-and-inventory-slope/tree/main/results">results</a> folder:

- df_slopes.xlsx: contains inventory linear regression's slope for each Product
- forecast_results.xlsx: contains forecast results for each Product for the next 12 weeks