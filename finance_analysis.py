#!/usr/bin/env python
# coding: utf-8

# # The Financial Instrument
# This Class analyses the financial instruments eg stocks etc. Any stock
# can be selected and can be analysed using the functions provided by the
# class or using pandas and numpy libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class FinancialInstrument():
    """
    A Class to analyze any Financial Instrument e.g. stock, etc.
    Uses yfinance to download the data for a stock, pandas and numpy to do arithmetic operations
    on it and matplotlib to visualize the stock data
    :param ticker: The ticker of the stock whose price shall be downloaded. e.g. for Infosys India it would be "INFY.NS" where INFY is the name of the stock and NS is for NSE stock market.
    :param start: The start date from which the stock prices shall be downloaded in format "YYYY-MM-DD"
    :param end: The end date till which the stock prices shall be downloaded in format "YYYY-MM-DD"
    """
    def __init__(self,ticker: str,start: str,end: str,interval: str):
        """

        :param ticker: The ticker of the stock whose price shall be downloaded. e.g. for Infosys India it would be "INFY.NS" where INFY is the name of the stock and NS is for NSE stock market.
        :param start: The start date from which the stock prices shall be downloaded in format "YYYY-MM-DD"
        :param end: The end date till which the stock prices shall be downloaded in format "YYYY-MM-DD"

        """
        self._ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.get_data()
        self.log_returns()

    def __repr__(self):
        return f"FinancialInstrument(ticker={self._ticker}, start={self.start},end={self.end})"

    def get_data(self):
        """
        Downloads data from yfinance
        :return: A dataframe with price and log_returns columns of the ticker the class is initialized with.
        """
        self.data = yf.download(self._ticker,self.start,self.end,interval=self.interval).Close.to_frame()
        self.data.rename(columns={'Close': 'price'},inplace = True)

    def log_returns(self):
        """
        Adds a column log_returns to self.data which contains the daily log_returns of the
        stock price.
        """
        self.data["log_returns"] = np.log(self.data.price/self.data.price.shift(1))

    def plot_prices(self):
        """
        Plots the line chart of the prices of stock
        """
        self.data.price.plot(figsize=(12,8))
        plt.title(f"Price Chart: {self._ticker}", fontsize=15)

    def plot_returns(self,kind='ts'):
        """
        Plots the returns for a given stock
        :param kind: 'ts' for line chart else 'hist' for histogram
        """

        if kind=='ts':
            self.data.log_returns.plot(figsize=(12,8))
            plt.title(f"Returns: {self._ticker}",fontsize=15)
        elif kind=='hist':
            self.data.log_returns.hist(figsize=(12,8),bins=int(np.sqrt(len(self.data))))
            plt.title(f"Frequency of Returns: {self._ticker}",fontsize=15)

    def set_ticker(self, ticker=None):
        """
        Changes the default ticker to new one
        :param ticker: Ticker of the stock to be changed
        """
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()

        else:
            print("Give the name of the ticker to change to.")

    def mean_returns(self,freq=None):
        """
        Calculates the mean returns of the returns
        :param freq: the frequency for stock price if it is None then the mean of all prices is calculated.
        :return: returns the mean of log returns of the stock price
        """
        if freq is None:
            return self.data.log_returns.mean()
        else:
            resampled_price = self.data.price.resample(freq).last()
            resampled_price = np.log(resampled_price / resampled_price.shift(1))
            return resampled_price.mean()

    def std_returns(self,freq=None):
        """
        Calculates the standard deviation / risk percentage of stock prices
        :param freq: the frequency for a stock price if it is None then the standard deviation of all prices is calculated
        else it is calculated for a stipulated freq value
        :return: returns the standard deviation of the stock price
        """
        if freq is None:
            return self.data.log_returns.std()
        else:
            resampled_price = self.data.price.resample(freq).last()
            resampled_price = np.log(resampled_price / resampled_price.shift(1))
            return resampled_price.std()


    def annualized_returns(self):
        mean_return = round(self.data.log_returns.mean() * 252, 3)
        risk_return = round(self.data.log_returns.std() * np.sqrt(252), 3)
        print(f"Returns : {mean_return} | Risk : {risk_return}")
