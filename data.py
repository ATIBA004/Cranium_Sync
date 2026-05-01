#this script imports data from yahoo finance
import yfinance as yf



symbol = ""
train_data_start_date = ""
train_data_end_date = ""
test_data_start_date = ""
test_data_end_date = ""

#this function downloads the training data from yahoo finance
def get_training_data(symbol, train_data_start_date, train_data_end_date):
    data = yf.download(symbol, start=train_data_start_date, end=train_data_end_date)
    return data

#this function download the test data from yahoo finance
def get_test_data(symbol, test_data_start_date, test_data_end_date):
    data = yf.download(symbol, start=test_data_start_date, end=test_data_end_date)
    return data



