# StockMarketPredictionUsingArimaModel
A Simple Stock Market Prediction using ARIMA &amp; LightGBM Model in Python

# Steps to run
1. Run fetchData.py file
* This code gets the data for all stocks from Yahoo Finance and saves it as Individual Files in a single csv file.
2. Run CleanData.py to perform cleanup on the data, and save it as individual csv files for each stock.
3. Run joinData.py to merge the cleanedup data of all stocks into a single file for training the model.
4. Run model.py to train the model using the data
5. Run predict.py to predict the next day return % of the stock
