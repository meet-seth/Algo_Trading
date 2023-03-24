from finance_analysis import FinancialInstrument as fi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

class Stock_Dataset(fi):
    def __init__(self,windows: list,horizons: list,intercross: bool,lookover: bool,**kwargs):
        """
        This class creates multiple datasets with varying windows and varying horizons so that multiple datasets
        can be created quickly.
        :param windows: List of windows to be produced. i.e if window is 7 then number of prices in X would be 7
        :param horizons: List of horizons to be produced. i.e. if horizon is 1 then Y will have price movement for 1 day.
        :param intercross: if true: eg. if windows = [5,7] and horizon = [2,3] then dataset produced would have
        [(5,2),(5,3),(7,2),(7,3)] windows and horizon pair respectively.
        else: the window horizon pair would be [(5,2),(7,3)]
        :param lookover: if true then single day windows would have prices for that day only not for the previous day where as otherwise
        they would have prices from previous day too. It is specifically for intra-day trading where we might not want previous day's prices to
        affect the model.
        :param kwargs: keyword arguments for Financial Instrument class which are ticker, start, end and interval specifically.
        """
        super().__init__(**kwargs)
        self.windows = windows
        self.horizons = horizons
        self.lookover = lookover
        self.intercross = intercross

    def create_dataset(self):
        self.dataset = {}
        if self.intercross:
            wh_list = list(product(self.windows,self.horizons))
        else:
            wh_list = [(i,j) for i,j in zip(self.windows,self.horizons)]

        for (i,j) in wh_list:
            tmp_dataset = self.create_windows_horizons(i,j)
            self.dataset[f'data_{i}_{j}'] = tmp_dataset


    def w_h(self,data,window,horizon):
        data['position'] = np.log(data.price / data.price.shift(horizon)).apply(lambda x: 1 if x>=0 else -1)
        data.dropna(inplace=True)
        for i in range(1,window+1):
            col = f't-{i}'
            data[col] = data.log_returns.shift(i)

        data.dropna(inplace=True)
        return data.copy()

    def create_windows_horizons(self,window,horizon):
        if self.lookover:
            grp = self.data.groupby(pd.Grouper(freq='D'))
            data_f = None
            for (t,val) in grp:
                val_2 = val.copy()
                tmp_data = self.w_h(val_2,window,horizon)
                if data_f is None:
                    data_f = tmp_data
                else:
                    data_f = pd.concat([data_f,tmp_data])

        else:
            data_f = self.w_h(self.data.copy(),window,horizon)

        return data_f


class Stock_Dataset_2(fi):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.dataset = None

    def create_dataset(self,qty=3,window=None,horizon=None):
        self.window=window
        self.horizon=horizon
        self.qty=qty
        signal_dict = {}
        for i in range(len(self.data)-self.horizon):
            price_on_day = self.data.price[i]
            for j in range(1,self.horizon+1):
                if f't+{j}' not in signal_dict.keys():
                    signal_dict[f't+{j}'] = []
                f_price = self.data.price.iloc[i+j]
                turnover = self.qty*(f_price+price_on_day)
                stt = round(0.1*turnover/100)
                txn_nse = round(0.00345*turnover/100,2)
                gst = round(18/100*txn_nse,2)
                stamp = round(0.015/100*qty*self.data.price.iloc[i])
                dp = 15.94
                tax = stt+txn_nse+gst+stamp+dp
                s_pnl = f_price - price_on_day
                if s_pnl>0:
                    pnl = s_pnl-tax
                    if pnl>0:
                        signal_dict[f't+{j}'].append(1)
                    else:
                        signal_dict[f't+{j}'].append(0)
                else:
                    pnl = s_pnl + tax
                    if pnl>0:
                        signal_dict[f't+{j}'].append(0)
                    else:
                        signal_dict[f't+{j}'].append(-1)


        for i in signal_dict.keys():
            self.data[i] = signal_dict[i]+[None]*self.horizon

        for i in range(1,self.window):
            self.data[f't-{i}'] = self.data.log_returns.shift(i)

        key_ls = []
        for k in range(-self.window+1,self.horizon+1):
            if k==0:
                key = 'log_returns'
            else:
                key = f't+{k}' if k>0 else f't{k}'
            key_ls.append(key)

        self.data[key_ls].dropna(inplace=True)

        new_targets = []
        target_ls = [f't+{i}' for i in range(1,self.horizon+1)]
        target_array = self.data[target_ls].to_numpy().astype(np.int32)
        for i in target_array:
            n_zero = 0
            for j in i:
                if j!=0:
                    n_zero=1
                    new_targets.append(j)
                    break
            if n_zero==0:
                new_targets.append(0)
        self.data['final_targets'] = new_targets

        key_ls = ['price'] + key_ls + ['final_targets']
        self.dataset = self.data[key_ls].dropna().copy()
        return self.dataset


