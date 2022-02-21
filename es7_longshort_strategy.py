from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction, DownImpulse
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, plot_pattern_m
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
from binance.client import Client
import datetime as dt
from tapy import Indicators
import time
import multiprocessing
from functools import partial
from ta import add_all_ta_features
from ta.utils import dropna
import random
from tqdm import tqdm
from scipy.signal import argrelextrema
import shutup; shutup.please()
from datetime import datetime
from typing import Union, Optional, Dict
import dateparser
import pytz
import json
from pybit import HTTP
import talib
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import time
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

load_model = True

skip = 1
layer_size = 500
output_size = 3
window_size = 20

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div


def get_state(parameters, t, window_size=20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d: t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0: t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


class Deep_Evolution_Strategy:
    inputs = None

    def __init__(
            self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=100, print_every=1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                        w
                        + self.learning_rate
                        / (self.population_size * self.sigma)
                        * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                if printout:
                    print(
                        'iter %d. reward: %f'
                        % (i + 1, self.reward_function(self.weights))
                    )
        if printout: print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.rand(input_size, layer_size)
            * np.sqrt(1 / (input_size + layer_size)),
            np.random.rand(layer_size, output_size)
            * np.sqrt(1 / (layer_size + output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, output_size)),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        decision = np.dot(feed, self.weights[1]) + self.weights[-1]
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend, minmax):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        self.minmax = minmax
        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        self._queue = []
        self._inventory = []

    def trade(self, data):
        """
        you need to make sure the data is [close, volume]
        """
        scaled_data = self.minmax.transform([data])[0]
        real_close = data[0]
        close = scaled_data[0]
        if len(self._queue) >= window_size:
            self._queue.pop(0)
        self._queue.append(scaled_data)
        if len(self._queue) < window_size:
            return {
                'status': 'data not enough to trade',
                'action': 'fail',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }
        state = self.get_state(
            window_size - 1,
            self._inventory,
            self._scaled_capital,
            timeseries = np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)
        if printout: print(prob)
        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            return {
                'status': 'buy 1 unit, cost %f' % (real_close),
                'action': 'buy',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }
        elif action == 2 and len(self._inventory):
            bought_price = self._inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            scaled_bought_price = self.minmax.inverse_transform(
                [[bought_price, 2]]
            )[0, 0]
            try:
                invest = (
                    (real_close - scaled_bought_price) / scaled_bought_price
                ) * 100
            except:
                invest = 0
            return {
                'status': 'sell 1 unit, price %f' % (real_close),
                'investment': invest,
                'gain': real_close - scaled_bought_price,
                'balance': self._capital,
                'action': 'sell',
                'timestamp': str(datetime.now()),
            }
        else:
            return {
                'status': 'do nothing',
                'action': 'nothing',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate(
            [state, [[len_inventory, z_inventory, z_capital]]], axis=1
        )
        return concat_parameters

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        invests = []
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                invests.append(invest)

            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )
        invests = np.mean(invests)
        if np.isnan(invests):
            invests = 0
        score = (starting_money - initial_money) / initial_money * 100
        return invests * 0.7 + score * 0.3

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        initial_money = self._scaled_capital
        starting_money = initial_money

        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        inventory = []
        real_inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            if printout: print(t, prob)

            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - window_size):
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                if printout:
                    print(
                        'day %d: buy 1 unit at price %f, total balance %f, total asset %f'
                        % (t, self.real_trend[t], real_starting_money, real_starting_money + sum(real_inventory))
                    )

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                real_bought_price = real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                states_sell.append(t)
                try:
                    invest = (
                                     (self.real_trend[t] - real_bought_price)
                                     / real_bought_price
                             ) * 100
                except:
                    invest = 0

                if printout:
                    print(
                        'day %d, sell 1 unit at price %f, investment %f %%, total balance %f, total asset %f'
                        % (t, self.real_trend[t], invest, real_starting_money, real_starting_money + sum(real_inventory))
                    )
            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )

        invest = (
                         (real_starting_money - real_initial_money) / real_initial_money
                 ) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest


session = HTTP(
    endpoint='https://api.bybit.com',
    api_key='o2aZhUESAachytlOy5',
    api_secret='AZPK3dhKdNsRhHX2s80KxCaEsFzlt5cCuQdK',
    spot=False
)

with open('config.json', 'r') as f:
    config = json.load(f)

exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
leverage = config['default']['leverage']

high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee = config['default']['fee']
fee_maker = config['default']['fee_maker']
fee_taker = config['default']['fee_taker']
fee_slippage = config['default']['fee_slippage']

period_days_ago = config['default']['period_days_ago']
period_days_ago_till = config['default']['period_days_ago_till']
period_interval = config['default']['period_interval']

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fcnt = config['default']['fcnt']
loop_count = config['default']['loop_count']


timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
condi_same_date = config['default']['condi_same_date']
long = config['default']['long']
o_fibo = config['default']['o_fibo']

profit_long = config['default']['profit_long']
profit_short = config['default']['profit_short']
stop_long = config['default']['stop_long']
stop_short = config['default']['stop_short']

symbol_random = config['default']['symbol_random']
symbol_duplicated = config['default']['symbol_duplicated']
symbol_last = config['default']['symbol_last']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']


intersect_idx = config['default']['intersect_idx']
plotview = config['default']['plotview']
printout = config['default']['printout']


def print_condition():
    print('-------------------------------')
    print('exchange:%s' % str(exchange))
    print('exchange_symbol:%s' % str(exchange_symbol))
    print('futures:%s' % str(futures))
    print('type:%s' % str(type))
    print('leverage:%s' % str(leverage))
    print('seed:%s' % str(seed))
    print('fee:%s%%' % str(fee*100))
    print('fee_maker:%s%%' % str(fee_maker*100))
    print('fee_taker:%s%%' % str(fee_taker*100))
    print('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
    if futures:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

    else:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

    print('timeframe: %s' % timeframe)
    print('period_days_ago: %s' % period_days_ago)
    print('period_days_ago_till: %s' % period_days_ago_till)
    print('period_interval: %s' % period_interval)
    print('round_trip_count: %s' % round_trip_count)
    print('compounding: %s' % compounding)
    print('fcnt: %s' % fcnt)
    print('loop_count: %s' % loop_count)

    print('symbol_duplicated: %s' % symbol_duplicated)
    print('symbol_random: %s' % symbol_random)
    print('symbol_last: %s' % symbol_last)
    print('symbol_length: %s' % symbol_length)

    # print('timeframe: %s' % timeframe)
    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    print('period: %s ~ %s' % (start_dt, end_dt))
    print('up_to_count: %s' % up_to_count)
    print('condi_same_date: %s' % condi_same_date)
    # print('long: %s' % long)
    print('o_fibo: %s' % o_fibo)

    print('profit_long: %s' % profit_long)
    print('profit_short: %s' % profit_short)
    print('stop_long: %s' % stop_long)
    print('stop_short: %s' % stop_short)

    print('intersect_idx: %s' % intersect_idx)
    print('plotview: %s' % plotview)
    print('printout: %s' % printout)
    print('-------------------------------')

client = None
if not futures:
    # basic
    client_basic = Client("basic_secret_key",
                    "basic_secret_value")
    client = client_basic
else:
    # futures
    client_futures = Client("futures_secret_key",
                    "futures_secret_value")
    client = client_futures


import threading
import functools
import time

def synchronized(wrapped):
    lock = threading.Lock()
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            # print ("Calling '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            result = wrapped(*args, **kwargs)
            # print ("Done '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            return result
    return _wrap



def date_to_milliseconds(date_str: str) -> int:
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    """
    # get epoch value in UTC
    epoch: datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d: Optional[datetime] = dateparser.parse(date_str, settings={'TIMEZONE': "UTC"})
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def get_historical_klines_pd(symbol, interval, start_date_str, end_date_str, start_int, end_int):
    """Get Historical Klines from Bybit
    See dateparse docs for valid start and end string formats
    http://dateparser.readthedocs.io/en/latest/
    If using offset strings for dates add "UTC" to date string
    e.g. "now UTC", "11 hours ago UTC"
    :param symbol: Name of symbol pair -- BTCUSD, ETCUSD, EOSUSD, XRPUSD
    :type symbol: str
    :param interval: Bybit Kline interval -- 1 3 5 15 30 60 120 240 360 720 "D" "M" "W" "Y"
    :type interval: str
    :param start_int: Start date string in UTC format
    :type start_int: str
    :param end_int: optional - end date string in UTC format
    :type end_int: str
    :return: list of OHLCV values
    """

    # set parameters for kline()
    timeframe = str(interval)
    start_ts = int(date_to_milliseconds(start_date_str)/1000)
    end_ts = None
    if end_date_str:
        end_ts = int(date_to_milliseconds(end_date_str)/1000)
    else:
        end_ts = int(date_to_milliseconds('now')/1000)

    # init our list
    output_data = []


    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    delta_seconds = (start_int - end_int) * 1440
    kline_loop_cnt = int(delta_seconds/200)
    kline_loop_last_limit = int(delta_seconds%200)

    symbol_existed = False
    # loop counter
    idx = 1
    limit = 200

    while True:
        # fetch the klines from start_ts up to max 200 entries
        temp_dict = session.query_mark_price_kline(
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            from_time=start_ts
        )

        # temp_dict = bybit.kline(symbol=symbol, interval=timeframe, _from=start_ts, limit=limit)
        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_dict):
            symbol_existed = True

        if symbol_existed:
            # extract data and convert to list
            temp_data = [list(i.values())[2:] for i in temp_dict['result']]
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            # NOTE: current implementation does not support inteval of D/W/M/Y
            start_ts = temp_data[len(temp_data) - 1][0] + interval*60

        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        # added start by neo
        if idx == kline_loop_cnt:
            limit = kline_loop_last_limit

        if idx > kline_loop_cnt:
            break

        # added end by neo

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API

        # if idx % 3 == 0:
        #     time.sleep(0.2)



    # convert to data frame
    df = pd.DataFrame(output_data, columns=['TimeStamp', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = [datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S.%d')[:-3] for i in df['TimeStamp']]
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    return df


@synchronized
def get_historical_ohlc_data_start_end(symbol, start_int, end_int, past_days=None, interval=None, futures=False):
    try:
        """Returns historcal klines from past for given symbol and interval
        past_days: how many days back one wants to download the data"""
        if not futures:
            # basic
            client_basic = Client("basic_secret_key",
                                  "basic_secret_value")
            client = client_basic
        else:
            # futures
            client_futures = Client("futures_secret_key",
                                    "futures_secret_value")
            client = client_futures

        if not interval:
            interval = '1h'  # default interval 1 hour
        if not past_days:
            past_days = 30  # default past days 30.

        start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).date())
        if end_int:
            end_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_int) + ' days')).date())
        else:
            end_date_str = None
        D = None
        try:
            if exchange_symbol == 'binance_futures':
                if futures:
                    D = pd.DataFrame(
                        client.futures_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))
                else:
                    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))

            elif exchange_symbol == 'bybit_usdt_perp':
                interval = int(interval[:1])
                D = get_historical_klines_pd(symbol, interval, start_date_str, end_date_str, start_int, end_int)
                return D, start_date_str, end_date_str

        except Exception as e:
            time.sleep(0.5)
            # print(e)
            return D, start_date_str, end_date_str

        if D is not None and D.empty:
            return D, start_date_str, end_date_str

        D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'is_best_match']
        D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
        D['symbol'] = symbol
        D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
               'taker_quote_vol']]
        D.rename(columns={
                            "open_date_time": "Date",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                          }, inplace=True)
        new_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        D['Date'] = D['Date'].astype(str)
        D['Open'] = D['Open'].astype(float)
        D['High'] = D['High'].astype(float)
        D['Low'] = D['Low'].astype(float)
        D['Close'] = D['Close'].astype(float)
        D['Volume'] = D['Volume'].astype(float)
        D = D[new_names]
    except Exception as e:
        # print('e in get_historical_ohlc_data_start_end:%s' % e)
        pass
    return D, start_date_str, end_date_str


def fractals_low_loopB(df, loop_count=1):
    for c in range(loop_count):
        i = Indicators(df)
        i.fractals()
        df = i.df
        df = df[~(df[['fractals_low']] == 0).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_low_loopA(df, fcnt=51, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fcnt] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df

def fractals_high_loopA(df, fcnt=5, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_high'] = df['High'].rolling(window, center=True).apply(lambda x: x[fcnt] == max(x), raw=True)
        df = df[~(df[['fractals_high']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high'], axis=1)
    return df

def sma(source, period):
    return pd.Series(source).rolling(period).mean().values

def backtest_tradeS7(df, symbol, df_lows_plot, df_highs_plot, trade_info):
    t = trade_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]


    dates_ = df.Date.tolist()
    closes_ = df.Close.tolist()
    sma7_ = sma(closes_, 7)
    # sma7_ = df.sma.tolist()

    high_ = df.High.tolist()
    low_ = df.Low.tolist()

    fee_maker_maker_percent = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    fee_maker_taker_slippage_percent = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)

    longshort = None
    position_enter_i = []
    position = False
    if closes_:
        for i, close in enumerate(closes_):
            if i < 1:
                continue

            sma7 = sma7_[:i+1]
            high = high_[:i+1]
            low = low_[:i+1]
            closes = closes_[:i+1]
            dates = dates_[:i+1]

            # long conditions
            c1 = True if sma7[-2] < sma7[-1] else False
            c2 = True if high[-2] > high[-1] else False
            c22 = True if low[-2] > low[-1] else False
            c3 = True #if closes[-1] <= sma7[-1] else False

            # short conditions
            c4 = True if sma7[-2] > sma7[-1] else False
            c55 = True if high[-2] < high[-1] else False
            c5 = True if low[-2] < low[-1] else False
            c6 = True #if closes[-1] >= sma7[-1] else False

            long_entry_condition = c1 and c2 and c22 and c3
            short_entry_condition = c4 and c5 and c55 and c6

            trends = df.High.tolist() if longshort else df.Low.tolist()
            detrends = df.Low.tolist() if longshort else df.High.tolist()

            s_profit = (position and trends[i] >= position_enter_i[-1] * (1 + profit_long)) if longshort else (
                        position and trends[i] <= position_enter_i[-1] * (1 - profit_short))
            s_stoploss = (position and detrends[i] < position_enter_i[-1] * (1 - stop_long)) if longshort else (
                        position and detrends[i] > position_enter_i[-1] * (1 + stop_short))

            if longshort is None and position is False and long_entry_condition:
                position = True
                position_enter_i = [dates[i], close]
                longshort = True
            elif longshort is None and position is False and short_entry_condition:
                position = True
                position_enter_i = [dates[i], close]
                longshort = False
            elif position is True:
                if s_profit or s_stoploss:
                    fee_percent = 0
                    pnl_percent = 0
                    trade_inout_i = []
                    if s_profit:
                        position_pf_i = [dates[i], position_enter_i[-1] * (1 + profit_long)]
                        pnl_percent = abs(profit_long) * leverage
                        fee_percent = fee_maker_maker_percent
                        trade_count.append(1)
                        trade_inout_i = [position_enter_i, position_pf_i]
                        order_history.append(trade_inout_i)
                        if printout: print('+ profit, ', i, close, position_enter_i, position_pf_i)

                    if s_stoploss:
                        position_sl_i = [dates[i], position_enter_i[-1] * (1 - stop_long)]
                        pnl_percent = -abs(stop_long) * leverage
                        fee_percent = fee_maker_taker_slippage_percent
                        trade_count.append(0)
                        trade_inout_i = [position_enter_i, position_sl_i]
                        order_history.append(trade_inout_i)
                        if printout: print('- stoploss, ', i, close, position_enter_i, position_sl_i)

                    asset_history_pre = asset_history[-1] if asset_history else seed
                    asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                    pnl_history.append(asset_history_pre*pnl_percent)
                    fee_history.append(asset_history_pre*fee_percent)
                    asset_history.append(asset_new)
                    # wavepattern_history.append(wavepattern)

                    winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                    trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history)]
                    stats_history.append(trade_stats)
                    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]

                    # if printout: print(str(trade_stats))
                    # print(symbol, fcnt, longshort, trade_inout_i[0][0][2:-3], trade_inout_i[0][1], '~', trade_inout_i[1][0][-8:-3], trade_inout_i[1][1], str(trade_stats))
                    print(symbol, fcnt, longshort, trade_inout_i[0][0][2:-3], str(trade_stats))
                    # print(symbol, trade_inut_i[0][0][2:-3], str(trade_stats))

                    if longshort is not None:
                        if plotview:
                            plot_pattern_m(df=df, wave_pattern=None, df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info, title=str(
                                symbol + ' %s '% str(longshort) + str(trade_stats)))

                    position = False
                    longshort = None
                    continue

    return trade_info


def check_has_same_wavepattern(symbol, fcnt, w_l, wavepattern):
    for wl in reversed(w_l):
        if symbol == wl[0] and fcnt == wl[1]:
            a0 = wl[-1].dates[0]
            b0 = wavepattern.dates[0]
            c0 = wl[-1].values[0]
            d0 = wavepattern.values[0]
            c0 = a0 == b0 and c0 == d0
            if condi_same_date:
                if a0 == b0:
                    return True

            a2 = wl[-1].dates[3]
            b2 = wavepattern.dates[3]
            c2 = wl[-1].values[3]
            d2 = wavepattern.values[3]
            c2 = a2 == b2 and c2 == d2

            a3 = wl[-1].dates[5]
            b3 = wavepattern.dates[5]
            c3 = wl[-1].values[5]
            d3 = wavepattern.values[5]
            c3 = a3 == b3 and c3 == d3

            a4 = wl[-1].dates[7]
            b4 = wavepattern.dates[7]
            c4 = wl[-1].values[7]
            d4 = wavepattern.values[7]
            c4 = a4 == b4 and c4 == d4

            a5 = wl[-1].dates[-1]
            b5 = wavepattern.dates[-1]
            c5 = wl[-1].values[-1]
            d5 = wavepattern.values[-1]
            c5 = a5 == b5 and c5 == d5

            if (c0 and c4) or (c4 and c5) or (c0 and c5) or (c0 and c3) or (c0 and c2 and c4) or (c0 and c2 and c3) or (c0 and c3 and c4):
                return True

            eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(wl[-1].dates))
            eq_values = np.array_equal(np.array(wavepattern.values), np.array(wl[-1].values))
            if eq_dates or eq_values:
                return True

    return False

wavepattern_l = list()

def sma_df(df, period=7):
    i = Indicators(df)
    i.sma(period=7)
    return i.df

agent = None

def loopsymbol(symbol, i, trade_info):

    past_days = 1
    timeunit = 'm'
    bin_size = str(timeframe) + timeunit
    start_int = i
    end_int = start_int - period_interval
    if end_int < 0:
        end_int = None
    df, start_date, end_date = get_historical_ohlc_data_start_end(symbol, start_int=start_int,
                                                                      end_int=end_int, past_days=past_days,
                                                                      interval=bin_size, futures=futures)
    # close = np.array(df[['Close']])
    rsi14 = talib.RSI(np.array(df['Close']), 14)
    macd, macdsignal, macdhist = talib.MACD(np.array(df['Close']), 12, 26, 9)
    df['rsi14'] = rsi14
    df['macd'] = macd
    # df_all = sma_df(df, period=7)
    if df is not None and not df.empty:

        # to get input_size
        # parameters = [df['Close'].tolist(), df['Volume'].tolist()]
        parameters = [df['macd'].tolist(), df['rsi14'].tolist()]
        inventory_size = 1
        mean_inventory = 0.5
        capital = 2
        concat_parameters = np.concatenate([get_state(parameters, 20), [[inventory_size,
                                                                         mean_inventory,
                                                                         capital]]], axis=1)

        input_size = concat_parameters.shape[1]


        # for real trade and fit
        real_trend = df['Close'].tolist()
        # parameters = [df['Close'].tolist(), df['Volume'].tolist()]
        parameters = [df['Close'].tolist(), df['Open'].tolist()]
        minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
        initial_money = np.max(parameters[0]) * 2
        import pickle
        file = 'model.pkl'
        model = None
        if os.path.isfile(file):
            with open(file, 'rb') as fopen:
                model = pickle.load(fopen)
                if printout: print('loaded model.pkl ')

        if not model:
            model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)

        agent = Agent(model=model,
                      timeseries=scaled_parameters,
                      skip=skip,
                      initial_money=initial_money,
                      real_trend=real_trend,
                      minmax=minmax)
        # if agent:
        # agent.change_data(timeseries=scaled_parameters,
        #                   skip=skip,
        #                   initial_money=initial_money,
        #                   real_trend=real_trend,
        #                   minmax=minmax)

        # if not trade_info[0]:
        #     print('learning symbol: %s' % symbol)
        #     agent.fit(iterations=100, checkpoint=10)


        states_buy, states_sell, total_gains, invest = agent.buy()

        fig = plt.figure(figsize=(15, 5))
        plt.plot(df['Close'], color='r', lw=2.)
        plt.plot(df['Close'], '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
        plt.plot(df['Close'], 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
        tt = '%s %s total gains %f, total investment %f%%' % (i, symbol, total_gains, invest)

        if not trade_info[0]:
            trade_info = list()


        print(tt)
        plt.title(tt)
        plt.legend()
        plt.show()

        agent.fit(iterations=100, checkpoint=10)

        # from datetime import datetime
        # volume = df['Volume'].tolist()
        # for i in range(100):
        #     print(agent.trade([real_trend[i], volume[i]]))
        import copy
        import pickle
        copy_model = copy.deepcopy(agent.model)
        with open('model.pkl', 'wb') as fopen:
            pickle.dump(copy_model, fopen)
        if printout: print('all done')






        # df_all = df
        # df_lows_plot = None
        # df_highs_plot = None
        # if not trade_info[0]:
        #     trade_info = backtest_tradeS7(df_all, symbol, df_lows_plot, df_highs_plot, trade_info)
    return trade_info


def single(symbols, i, trade_info, *args):
    for symbol in symbols:
        loopsymbol(symbol, i, trade_info)
    return trade_info


def round_trip(i):
    round_trip = []
    symbols_futures = [
                        # 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                        'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VETUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT', 'MATICUSDT', 'ATOMUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'DOGEUSDT', 'ANKRUSDT', 'MTLUSDT', 'TOMOUSDT', 'DENTUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT', 'NKNUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'BCHUSDT', 'OGNUSDT', 'BTSUSDT', 'COTIUSDT', 'SOLUSDT', 'CTSIUSDT', 'CHRUSDT', 'LENDUSDT', 'STMXUSDT', 'KNCUSDT', 'LRCUSDT', 'COMPUSDT', 'SCUSDT', 'ZENUSDT', 'SNXUSDT', 'DGBUSDT', 'SXPUSDT', 'MKRUSDT', 'STORJUSDT', 'MANAUSDT', 'YFIUSDT', 'BALUSDT', 'BLZUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT', 'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'RUNEUSDT', 'BELUSDT', 'UNIUSDT', 'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'AUDIOUSDT', 'CTKUSDT', 'AKROUSDT', 'AXSUSDT', 'UNFIUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'REEFUSDT', 'CELOUSDT', 'BTCSTUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'ALICEUSDT', 'LINAUSDT', 'TLMUSDT', 'BAKEUSDT', 'ICPUSDT', 'ARUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'ATAUSDT', 'GTCUSDT', 'KEEPUSDT', 'KLAYUSDT', 'C98USDT', 'RAYUSDT', 'DYDXUSDT', 'GALAUSDT', 'ENSUSDT', 'PEOPLEUSDT']

    # symbols_futures = ['FLMUSDT']
    asset_history = [seed]
    trade_count = []
    fee_history = []
    pnl_history = []



    symbols = symbols_futures

    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]
    print(symbols)

    # single
    single(symbols, i, trade_info)
    round_trip.append(trade_info[2][-1])
    print(i, ' | ', trade_info[2][-1], len(trade_count), ' | ', asset_history[-1])
    return round_trip

if __name__ == '__main__':

    print_condition()
    rount_trip_total = []
    start = time.perf_counter()
    if round_trip_flg:
        for i in range(round_trip_count):
            rt = list()
            try:
                start = time.perf_counter()
                # cpucount = multiprocessing.cpu_count() * 1
                # cpucount = multiprocessing.cpu_count()
                cpucount = 1
                print('cpucount:%s' % cpucount)
                pool = multiprocessing.Pool(processes=cpucount)
                rt = pool.map(round_trip, range(period_days_ago, period_days_ago_till, -1 * period_interval))
                pool.close()
                pool.join()
                start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
                end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
                print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
            except Exception as e:
                # print(e)
                pass

            print('============ %s stat.==========' % str(i))
            r = list(map(lambda i: i[0], rt))

            winrate_l = list(map(lambda i: 1 if i[0] > seed else 0, rt))
            meanaverage = round((sum(r)/len(r)), 2)
            roundcount = len(rt)
            winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
            total_gains = (meanaverage - seed)*roundcount
            print('round r: %s' % r)
            print('round winrate_l: %s' % str(winrate_l))
            print('round roundcount: %s' % roundcount)
            print('round winrate: %s' % winrate)
            print('round meanaverage: %s' % str(meanaverage))
            print('round total gains: %s' % str(total_gains))
            print('============ %s End All=========='% str(i))

            rount_trip_total.append([meanaverage, roundcount, winrate, total_gains])
        print_condition()
        for i, v in enumerate(rount_trip_total):
            print(i, v)
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    else:

        symbols = list()

        ## binance symbols
        symbols_futures = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT',
            'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VETUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'HOTUSDT', 'ZILUSDT',
            'ZRXUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'OMGUSDT', 'THETAUSDT',
            'ENJUSDT', 'MATICUSDT', 'ATOMUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'DOGEUSDT', 'ANKRUSDT', 'MTLUSDT',
            'TOMOUSDT', 'DENTUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT',
            'NKNUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'BCHUSDT', 'OGNUSDT', 'BTSUSDT', 'COTIUSDT',
            'SOLUSDT', 'CTSIUSDT', 'CHRUSDT', 'LENDUSDT', 'STMXUSDT', 'KNCUSDT', 'LRCUSDT', 'COMPUSDT', 'SCUSDT',
            'ZENUSDT', 'SNXUSDT', 'DGBUSDT', 'SXPUSDT', 'MKRUSDT', 'STORJUSDT', 'MANAUSDT', 'YFIUSDT', 'BALUSDT',
            'BLZUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT',
            'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'RUNEUSDT', 'BELUSDT', 'UNIUSDT',
            'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'AUDIOUSDT', 'CTKUSDT',
            'AKROUSDT', 'AXSUSDT', 'UNFIUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'REEFUSDT', 'CELOUSDT',
            'BTCSTUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'ALICEUSDT', 'LINAUSDT', 'TLMUSDT', 'BAKEUSDT', 'ICPUSDT',
            'ARUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'ATAUSDT', 'GTCUSDT', 'KEEPUSDT', 'KLAYUSDT', 'C98USDT',
            'RAYUSDT', 'DYDXUSDT', 'GALAUSDT', 'ENSUSDT', 'PEOPLEUSDT']


        ## bybit symbols
        symbols_bybit = []
        symbols_info = session.query_symbol()
        if symbols_info['ret_msg'] == 'OK':
            symbol_result = symbols_info['result']
            for sym in symbol_result:
                quote_currenty = sym['quote_currency']
                if quote_currenty == 'USDT':
                    symbol_name = sym['name']
                    symbols.append(symbol_name)
            symbols_bybit = symbols

        if symbol_duplicated:
            symbols = [x for x in symbols_futures if x in symbols_bybit]
        elif exchange == 'bybit_usdt_perp':
            symbols = symbols_bybit
        elif exchange == 'binance_futures':
            symbols = symbols_futures


        if symbol_last:
            symbols = symbols[symbol_last:]
        if symbol_length:
            symbols = symbols[:symbol_length]
        if symbol_random:
            symbols = random.sample(symbols, len(symbols))

        print(len(symbols), symbols)

        # symbols = ['LUNAUSDT']
        # print(len(symbols), symbols)

        stats_history = []
        order_history = []
        asset_history = []
        trade_count = []
        fee_history = []
        pnl_history = []
        wavepattern_history = []
        trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]
        r = range(period_days_ago, period_days_ago_till, -1 * period_interval)
        for i in r:
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            single(symbols, i, trade_info)
            if trade_info[2]:
                print(str(i)+'/'+str(len(r)), ' now asset: ', trade_info[2][-1], ' | ', len(trade_count), ' | pre seed: ', asset_history_pre)
            else:
                print(str(i)+'/'+str(len(r)), ' now asset: ', seed, ' | ', len(trade_count), ' | pre seed: ', seed)

        print('============ %s stat.==========' % str(i))
        winrate_l = list(map(lambda i: 1 if i > 0 else 0, pnl_history))
        meanaverage = round((sum(asset_history)/len(asset_history)), 2) if asset_history and len(asset_history) > 0 else None
        roundcount = len(trade_count)
        winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2)) if winrate_l and len(winrate_l) > 0 else None
        print('round r: %s' % roundcount)
        print('round winrate_l: %s' % str(winrate_l))
        print('round roundcount: %s' % roundcount)
        print('round winrate: %s' % winrate)
        print('round meanaverage: %s' % str(meanaverage))
        print('round total gains: %s' % str(trade_info[-2][-1] if trade_info[2] else 0))
        print('============ %s End All=========='% str(i))
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    print("good luck done!!")