# this script imports data from yahoo finance
import yfinance as yf
import numpy as np


# this is the symbol and the times we want to downlaod the data for
symbol = "VCM.TO"
train_data_start_date = "2005-01-01"
train_data_end_date = "2020-01-01"
test_data_start_date = ""
test_data_end_date = ""

data = yf.download(symbol, start=train_data_start_date, end=train_data_end_date)



def add_relevant_features(data):
    # close price of the previous day
    data["prev_close"] = data["Close"].shift(1)

    # open price of the previous day
    data["prev_open"] = data["Open"].shift(1)

    # high of the previous day
    data["prev_high"] = data["High"].shift(1)

    # low of the previous day
    data["prev_low"] = data["Low"].shift(1)

    # since we will use a regression model,
    # we will use logarithmic returns
    data["1d_return"] = np.log( data["Close"] / data["prev_close"] )    #1 day return

    # volatility of the previous 5 days
    data["vol_5d"] = data["1d_return"].rolling(window=5).std()

    # volatility of the previous 10 days
    data["vol_10d"] = data["1d_return"].rolling(window=10).std()

    # volatility of the previous 20 days
    data["vol_20d"] = data["1d_return"].rolling(window=20).std()

    
    return data




def get_training_data(symbol, train_data_start_date, train_data_end_date):
    """
    This function downloads the training data from yahoo finance
    and adds the relevant features to the data

    params: symbol: the symbol of the stock to download the data for
            train_data_start_date: the start date of the training data
            train_data_end_date: the end date of the training data

    returns: the training data with the relevant features
    """

    data = yf.download(symbol, start=train_data_start_date, end=train_data_end_date)
    data.columns = data.columns.get_level_values(0)
    data = add_relevant_features(data)
    data = data.dropna()
    return data


def get_test_data(symbol, test_data_start_date, test_data_end_date):
    """
    This function downloads the test data from yahoo finance
    and adds the relevant features to the data

    params: symbol: the symbol of the stock to download the data for
            test_data_start_date: the start date of the test data
            test_data_end_date: the end date of the test data

    returns: the test data with the relevant features
    """

    data = yf.download(symbol, start=test_data_start_date, end=test_data_end_date)
    data.columns = data.columns.get_level_values(0)
    data = add_relevant_features(data)
    data = data.dropna()
    return data

get_training_data(symbol, train_data_start_date, train_data_end_date).to_csv("training_data.csv")


