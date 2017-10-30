import os, random

import numpy as np
import pandas as pd


def pctrank(data):
    return pd.Series(data).rank(pct=True).iloc[-1]


def get_csv_file(days=252, datadir="", filename=""):
    try_random = 10
    name = None
    data = None

    while try_random:


        if not datadir or not os.path.isdir(datadir):
            raise ValueError("Incorrect CSV folder")

        if not filename:
            filename = random.choice(os.listdir(datadir))
        csv_file = os.path.join(datadir, filename)

        # self.datadir = datadir  # os.path.join('Data', 'daily_sp500_1998-2013')
        # self.dataset = {}
        #
        # for csv_file in glob.glob(os.path.join(datadir, '*.csv')):
        #     data = pd.read_csv(csv_file, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
        #                        parse_dates=['Date'])
        #     # data = data.sort_values(by='Date')
        #     data.set_index('Date', inplace=True)
        #     name = os.path.splitext(os.path.basename(csv_file))[0]
        #     self.dataset[name] = data

        if not os.path.isfile(csv_file):
            raise ValueError("Incorrect CSV input file")

        data = pd.read_csv(csv_file, header=None, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                           parse_dates=['Date'])
        name = os.path.splitext(os.path.basename(csv_file))[0]

        if data.shape[0] > days + 30:
            break

        try_random -= 1
        name = None
        data = None

    return name, data


class MarketData(object):
    MinPercentileDays = 100

    def __init__(self, days=252, datadir="", filename="", scale=True):
        name, data = get_csv_file(days=days, datadir=datadir, filename=filename)

        print("Load file : {}".format(name))
        if not name:
            raise ValueError('Cannot get data from CSV')

        self.days = days + 1
        self.idx = 0

        # data = data.sort_values(by='Date')
        data.set_index('Date', inplace=True)
        del data['Time']
        # clear nan and 0 volume
        data = data[~np.isnan(data.Volume)]
        data.Volume.replace(0, 1, inplace=True)
        data['Change'] = (data.Close - data.Close.shift()) / data.Close.shift()
        pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        # data['ClosePctl'] = data.Close.expanding(self.MinPercentileDays).apply(pctrank)
        # data['VolumePctl'] = data.Volume.expanding(self.MinPercentileDays).apply(pctrank)
        # Scale
        Change = data.Change
        if scale:
            mean_values = data.mean(axis=0)
            std_values = data.std(axis=0)
            data = (data - np.array(mean_values)) / np.array(std_values)
        data['Change'] = Change

        # move change column to index front
        cols = data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Change')))
        data = data.reindex(columns=cols)

        self.min_values = data.min(axis=0)
        self.max_values = data.max(axis=0)
        self.data = data
        self.step = 0

    def env_info(self):
        print('From : ', self.data.iloc[self.idx].name)
        print('To   : ', self.data.iloc[self.idx + self.days].name)
        print(self.data.iloc[self.idx:self.idx + self.days])

    def reset(self):
        # we want contiguous data
        self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days)
        self.step = 0

    def _step(self):
        obs = self.data.iloc[self.idx].as_matrix()
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done
