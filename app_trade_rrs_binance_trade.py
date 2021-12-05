import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math
from datetime import datetime
from tqdm import tqdm
import random
import statistics
import sys
# sys.stdout = open('app_trade_rrs_binance_out', 'a')
plt.rc('font', size=18)        # 기본 폰트 크기
plt.rc('axes', labelsize=18)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=18)  # x축 눈금 폰트 크기
plt.rc('ytick', labelsize=18)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=14)  # 범례 폰트 크기
plt.rc('figure', titlesize=20) # figure title 폰트 크기
def get_historical_ohlc_data(symbol, past_days=None, interval=None):
    """Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data"""
    from binance.client import Client
    # matplotlib.use('TkAgg')
    # sns.set()

    client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq",
                    "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
    if not interval:
        interval = '1h'  # default interval 1 hour
    if not past_days:
        past_days = 30  # default past days 30.

    start_str = str((pd.to_datetime('today') - pd.Timedelta(str(past_days) + ' days')).date())

    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, interval=interval))
    D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                 'taker_base_vol', 'taker_quote_vol', 'is_best_match']
    D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
    D['symbol'] = symbol
    D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
           'taker_quote_vol']]

    return D

# https://tcoil.info/compute-rsi-for-stocks-with-python-relative-strength-index/
def indi_rsi(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def indi_wrplus(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    wr += 100
    return wr

# Fast %K = ((현재가 - n기간 중 최저가) / (n기간 중 최고가 - n기간 중 최저가)) * 100
def get_stochastic_fast_k(close_price, low, high, n=5):
    fast_k = ((close_price - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return fast_k

# Slow %K = Fast %K의 m기간 이동평균(SMA)
def get_stochastic_slow_k(fast_k, n=3):
    slow_k = fast_k.rolling(n).mean()
    return slow_k

# Slow %D = Slow %K의 t기간 이동평균(SMA)
def get_stochastic_slow_d(slow_k, n=3):
    slow_d = slow_k.rolling(n).mean()
    return slow_d

wrp_high_limit = 80
wrp_low_limit = 20
rsi_high_limit = 70
rsi_low_limit = 30

def strategy_entry_condition2(wrp_l, rsi_l):
    signal = False
    # 0:ex, 1:now
    if wrp_l[0] > wrp_high_limit and wrp_l[1] < wrp_high_limit \
            and (rsi_l[1] > rsi_high_limit or rsi_l[0] > rsi_high_limit):
        signal = True
    elif wrp_l[0] < wrp_low_limit and wrp_l[1] < wrp_low_limit \
            and (rsi_l[1] < rsi_low_limit or rsi_l[0] < rsi_low_limit):
        signal = True
    return signal

def strategy_entry_condition(wrp, rsi):
    signal = False
    # 0:ex, 1:now
    if wrp and rsi:
        if wrp > wrp_high_limit and rsi > rsi_high_limit:
            signal = True
        elif wrp < wrp_low_limit and rsi < rsi_low_limit:
            signal = True
    return signal






def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def get_state(parameters, t, window_size = 20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d : t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0 : t + 1]
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
    
    def train(self, epoch = 100, print_every = 1):
        # lasttime = time.time()
        reward_avg = []
        # for i in tqdm(range(epoch), desc = "train processing epoch", mininterval = 10):
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
                reward_avg.append(self.reward_function(self.weights))
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('reward_avg:%s \n' % str(float(sum(reward_avg))/float(len(reward_avg))))
        # print('time taken to train:', time.time() - lasttime, 'seconds')
        
class Model:
    def __init__(self, input_size, layer_size, output_size):
    #     self.weights = [
    #         np.random.rand(input_size, layer_size)
    #         * np.sqrt(1 / (input_size + layer_size)),
    #         np.random.rand(layer_size, output_size)
    #         * np.sqrt(1 / (layer_size + output_size)),
    #         np.zeros((1, layer_size)),
    #         np.zeros((1, output_size)),
    #     ]
    #
    # def predict(self, inputs):
    #     feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
    #     # decision = np.dot(feed, self.weights[1]) + self.weights[-1]
    #     decision = np.dot(feed, self.weights[1])
    #     return decision

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



    #     self.weights = [
    #         np.random.randn(input_size, layer_size),
    #         np.random.randn(layer_size, output_size),
    #         np.random.randn(layer_size, 1),
    #         np.random.randn(1, layer_size),
    #     ]
    #
    # def predict(self, inputs):
    #     feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
    #     decision = np.dot(feed, self.weights[1])
    #     buy = np.dot(feed, self.weights[2])
    #     return decision, buy



    #     self.weights = [
    #         # np.random.randn(input_size, layer_size),
    #         # np.random.randn(layer_size, output_size),
    #         np.random.rand(input_size, layer_size) * np.sqrt(1 / (input_size + layer_size)),
    #         np.random.rand(layer_size, output_size) * np.sqrt(1 / (layer_size + output_size)),
    #         np.random.randn(layer_size, buy_sell_count_max),
    #         np.random.randn(1, layer_size),
    #     ]
    #
    # def predict(self, inputs):
    #     feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
    #     decision = np.dot(feed, self.weights[1])
    #     # buy = np.dot(feed, self.weights[2])
    #     buy = 1
    #     return decision, buy

        # self.weights = [
        #     np.random.rand(input_size, layer_size) * np.sqrt(1 / (input_size + layer_size)),
        #     np.random.rand(layer_size, output_size) * np.sqrt(1 / (layer_size + output_size)),
        #     np.zeros((1, layer_size)),
        #     np.zeros((1, output_size)),
        # ]

    # def predict(self, inputs):
    #     feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
    #     decision = np.dot(feed, self.weights[1]) + self.weights[-1]
    #     buy = 1
    #     return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend, real_df, minmax):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.real_df = real_df
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
        self._hold_size = 0

        self.positions = []
        self.realized_pnl_total = 0
        self.wallet_balance_t = []
        self.fee_total = 0

        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        # if len(parameters) == 2:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        # elif len(parameters) == 3:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0]])[0, 0]
        # elif len(parameters) == 4:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0, 0]])[0, 0]
        # elif len(parameters) == 5:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0, 0, 0]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        # if len(parameters) == 2:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        # elif len(parameters) == 3:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0]])[0, 0]
        # elif len(parameters) == 4:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0, 0]])[0, 0]
        # elif len(parameters) == 5:
        #     self._scaled_capital = self.minmax.transform([[self._capital, 2, 0, 0, 0]])[0, 0]
        self._queue = []
        self._inventory = []
        self._hold_size = []



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
            self._hold_size,
            self._inventory,
            self._scaled_capital,
            timeseries = np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)
        # action, prob = self.act(state)
        buy = 1
        # action, buy = self.act(state)
        holdsize, roe, pnl_roe, quantity_limit, transac_size = self.actionlogic(action, buy, real_close)

        # for next state
        self._hold_size = holdsize
        self._inventory = roe
        self._scaled_capital = pnl_roe

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

    def change_data(self, timeseries, skip, initial_money, real_trend, real_df, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.real_df = real_df
        self.minmax = minmax
        self._initiate()

    # def act(self, sequence):
    #     decision, buy = self.model.predict(np.array(sequence))
    #     return np.argmax(decision[0]), np.argmax(buy[0]) + 1

    # def act(self, sequence):
    #     decision = self.model.predict(np.array(sequence))
    #     return np.argmax(decision[0]), 1


    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, holdsize, roe, pnl_roe, timeseries):
        state = get_state(timeseries, t)
        concat_parameters = np.concatenate(
            [state, [[holdsize, roe, pnl_roe]]], axis = 1
        )
        return concat_parameters

    def actionlogic(self, action, buy, price):
        # futures
        acts = []
        charts = []
        trades = []
        positions = []
        orders = []
        assets = []
        fee_total = 0
        realized_pnl_total = 0
        fee_rate = 0.04
        price_ex = 0
        dif_price = 0
        dif_percent = 0
        states_buy = []
        states_sell = []
        net_pnl_t = 0

        wallet_balance = 0
        wallet_balance_t = []
        holdsize = 0
        roe = 0
        pnl_roe = 0
        pnl_roes = []
        quantity_limit = 0
        transac_size = 0

        if action == 1 or action == 2:
            entry = True
            # if flg == 'buy':
            # wrp = self.real_df.iloc[t]['wrplus']
            # rsi = self.real_df.iloc[t]['rsi']
            # entry = strategy_entry_condition(wrp, rsi)
            # entry = True
            # wrp1 = self.real_df.iloc[t]['wrplus']
            # wrp0 = self.real_df.iloc[t-1]['wrplus']
            # rsi1 = self.real_df.iloc[t]['rsi']
            # rsi0 = self.real_df.iloc[t-1]['rsi']
            # entry = strategy_entry_condition2([wrp0, wrp1], [rsi0, rsi1])
            if entry:
                last_avg_price = self.positions[-1][0] if self.positions else 0
                last_hold_size = self.positions[-1][1] if self.positions else 0

                di = 1 if action == 1 else -1
                buy_sell = 'buy' if action == 1 else 'sell'

                quantity = buy
                quantity_limit = quantity
                if (abs(last_hold_size + di * quantity)) > buy_sell_count_max:
                    quantity_limit = buy_sell_count_max - abs(last_hold_size)

                # futures
                # time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                # symbol = 'BTCUSDT'
                transac_size = di * quantity_limit
                fee = -abs(transac_size * fee_rate * price * 1 / 100)

                # order
                order = [price, transac_size, buy_sell]
                holdsize = last_hold_size + transac_size
                realized_pnl = 0
                if last_hold_size * transac_size > 0:
                    avg_price = abs(
                        (last_avg_price * last_hold_size + price * transac_size) / holdsize) if holdsize else 0
                    realized_pnl = 0
                else:
                    if abs(last_hold_size) == abs(transac_size):
                        avg_price = 0
                        realized_pnl = (price - last_avg_price) * last_hold_size
                    elif abs(last_hold_size) > abs(transac_size):
                        avg_price = last_avg_price
                        realized_pnl = (last_avg_price - price) * transac_size

                    elif abs(last_hold_size) < abs(transac_size):
                        avg_price = price
                        realized_pnl = (price - last_avg_price) * last_hold_size

                pnl = (price - last_avg_price) * last_hold_size if last_avg_price else 0
                roe = round((pnl / abs(last_avg_price * last_hold_size)) * 100,
                            2) if last_avg_price * last_hold_size else 0
                self.realized_pnl_total += realized_pnl

                # trade
                trade = [price, transac_size, buy_sell, fee, realized_pnl]
                # trades.clear()
                # trades.append(trade)

                # position
                position = [avg_price, holdsize, pnl, roe]
                self.positions.clear()
                self.positions.append(position)

                self.fee_total += fee
                wallet_balance = initial_money + self.realized_pnl_total + self.fee_total + pnl
                net_pnl_t = self.realized_pnl_total + self.fee_total + pnl

                # asset
                invest_money = abs(avg_price * holdsize)
                cash_money = initial_money + self.realized_pnl_total + self.fee_total - invest_money
                asset = [initial_money, wallet_balance, invest_money, cash_money]
                self.wallet_balance_t.append(wallet_balance)

                # analysis
                analysis = [net_pnl_t, round((net_pnl_t / initial_money) * 100, 4), self.fee_total, self.realized_pnl_total]
                net_pnl_t = self.realized_pnl_total + self.fee_total + pnl
                pnl_roe = (net_pnl_t / initial_money) * 100
                # assets.append(asset)

        return holdsize, roe, pnl_roe, quantity_limit, transac_size

    def get_pnl(self, flg=None):
        # initial_money = self._scaled_capital
        initial_money = self.initial_money
        state = self.get_state(0, 0, 0, 0, self.timeseries)

        # futures
        acts = []
        charts = []
        trades = []
        positions = []
        orders = []
        assets = []
        fee_total = 0
        realized_pnl_total = 0
        print_flg = False
        if flg == 'buy':
            print_flg = True
        fee_rate = 0.04
        price_ex = 0
        dif_price = 0
        dif_percent = 0
        states_buy = []
        states_sell = []
        net_pnl_t = 0

        wallet_balance = 0
        wallet_balance_t = []
        holdsize = 0
        roe = 0
        pnl_roe = 0
        pnl_roes = []
        quantity_limit = 0
        transac_size = 0

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            # action, prob = self.act_softmax(state)
            # print(prob)
            buy = 1
            # act
            act = [t, action, buy]
            price = self.real_trend[t]
            if t > 0:
                price_ex = self.real_trend[t - 1]
                dif_price = price - price_ex
                dif_percent = round(dif_price/100, 4)

            holdsize, roe, pnl_roe, quantity_limit, transac_size = self.actionlogic(action, buy, price)

            if flg == 'buy':
                if action == 1 and quantity_limit > 0 and transac_size:
                    states_buy.append(t)
                elif action == 2 and quantity_limit > 0 and transac_size:
                    states_sell.append(t)

            state = self.get_state(t + 1, holdsize, roe, pnl_roe, self.timeseries)
            pnl_roes.append(pnl_roe)

        roe_total = pnl_roe

        if flg == 'get_reward':
            return roe_total
        else:
            total_result_l.append(round(roe_total, 2))
            print('[%s] roe_total: %f%%, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' %
                  (symbol, round(roe_total, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   len(states_buy) + len(states_sell)))

            print('plot start')

            plt.figure(figsize=(20, 20))
            gs = gridspec.GridSpec(nrows=3,
                                   ncols=1,
                                   height_ratios=[10, 3, 2]
                                   )

            # 1st
            ax0 = plt.subplot(gs[0])
            label = '[%s] wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' % \
                    (symbol, wallet_balance, initial_money, len(states_buy), len(states_sell),
                   str(len(states_buy) + len(states_sell)))

            plt.title(label)
            plt.xlabel(fit_i_c + '  ' + interval_str + '  '
                       + str(self.real_df['open_date_time'].iloc(0)[0]) + ' ~ ' + str(self.real_df['open_date_time'].iloc(0)[-1]))
            plt.ylabel(symbol)
            ax0.plot(self.real_trend, label='true close', c='b')
            ax0.plot(
                self.real_trend, 'o', label='predict buy', markevery=states_buy, c='g'
            )
            ax0.plot(
                self.real_trend, 'o', label='predict sell', markevery=states_sell, c='r'
            )
            ax0.legend()



            # 2nd
            ax1 = plt.subplot(gs[1])
            ax1.plot(pnl_roes, label='pnl_roe', c='g')
            ax1.legend()

            label = 'roe_total: %f%%, ' % \
                    (round(roe_total, 2))
            plt.title(label)
            # plt.xlabel(fit_i_c + '  ' + interval_str + '  '
            #            + str(self.real_df['open_date_time'].iloc(0)) + ':' + str(self.real_df['open_date_time'].iloc(-1)))
            plt.ylabel(symbol)

            # 3rd
            ax2 = plt.subplot(gs[2])
            ax2.plot(self.real_df['rsi'], color='green', linewidth=2)
            ax2.axhline(wrp_high_limit, linewidth=1.5, linestyle='--', color='grey')
            ax2.axhline(wrp_low_limit, linewidth=1.5, linestyle='--', color='grey')
            plt.tight_layout()
            plt.savefig('../xyz_svg/%s+%s+%s.svg' % (symbol, interval_str, datetime.now().strftime("%m:%d:%Y_%H:%M:%S")))
            plt.show()
            print('plot end')
            print('total_result average:%f' % statistics.mean(total_result_l))
            print('total_result:\n%s' % str(total_result_l))


    def get_reward(self, weights):
        self.model.weights = weights
        return self.get_pnl(flg='get_reward')


    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def buy(self):
        return self.get_pnl(flg='buy')


###################
## symbol setting ##
###################
from binance.client import Client
client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq", "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
exchange_info = client.get_exchange_info()
symbols = []
for s in exchange_info['symbols']:
    symbol = s['symbol']
    symbols.append(symbol)
# print(symbols)

find_USDT = 'USDT'
index_USDTs = [i for i in range(len(symbols)) if find_USDT in symbols[i]]

symbols_usdt = []
for s in index_USDTs:
    symbols_usdt.append(symbols[s])
# print(symbols_usdt)
# print(len(symbols_usdt))

total_result_l = []
use_file_model = True
print('use_file_model:%s' % str(use_file_model))

###################
## data settting ##
###################
# 1day:1440m
# symbols_usdt = ['AXSUSDT']
symbols_usdt = random.sample(symbols_usdt, len(symbols_usdt))
print(symbols_usdt)

for symbol in symbols_usdt:
    past_days = 2
    interval = 15  # 1, 5, 15
    timeunit = 'm'
    interval_str = str(interval) + timeunit
    stick_cnt_per_day = past_days * 1440 / interval if timeunit == 'm' else 0
    divide = 4
    period_unit = math.floor(stick_cnt_per_day/divide)
    print('past_days:%s interval_str:%s period_unit:%s (total_unit:%s)' % (past_days, interval_str, period_unit, stick_cnt_per_day))

    print('buy symbol %s' % (symbol))
    try:
        df = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)
    except Exception as e:
        print('get_historical_ohlc_data: %s' % (e))
        continue
    df = df[['open_date_time', 'open', 'high', 'low', 'close', 'volume']]
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['open'] = df['volume'].astype(float)
    df['high'] = df['volume'].astype(float)
    df['low'] = df['volume'].astype(float)
    df['rsi'] = indi_rsi(df['close'], 14)
    df['wrplus'] = indi_wrplus(df['high'], df['low'], df['close'], 14)
    df['fast_k'] = get_stochastic_fast_k(df['close'], df['low'], df['high'], 14)
    df['slow_k'] = get_stochastic_slow_k(df['fast_k'], 14)
    df['slow_d'] = get_stochastic_slow_d(df['slow_k'], 14)
    df = df.fillna(0)
    print('len(df):%s' % len(df))
    # print(df.head())

    ## train data dft
    dft = df.iloc[np.r_[0:period_unit]]
    print('======================== %s agent running ...  ============' % (symbol))
    parameters = [dft['close'].tolist(), dft['volume'].tolist()]

    ################
    # model setting
    ################
    # fit prod
    # iter = 500
    # check = 20

    # # fit dev
    # iter = 100
    # check = 20
    #
    # fit debug
    iter = 50
    check = 10

    fit_i_c = 'fit_i_c:%s_%s' % (iter, check)
    print(fit_i_c)

    window_size = 20
    skip = 1
    layer_size = 500
    output_size = 3

    holdsize = 0
    roe = 0.0
    pnl_roe = 0.0

    buy_sell_count_max = 1

    concat_parameters = np.concatenate([get_state(parameters, window_size), [[holdsize, roe, pnl_roe]]], axis=1)
    input_size = concat_parameters.shape[1]

    try:
        if use_file_model:
            with open('app_trade_rrs_binance_trade.pkl', 'rb') as fopen:
                model = pickle.load(fopen)
                print('\n======================== Use [app_trade_rrs_binance_trade.pkl] ====================')
        else:
            model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)
            print('\n======================== Use new model =====================\n')
    except Exception:
        model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)
        print('\n======================== Use new model =====================\n')


    ###############################
    # agent fit  (train 4 : buy 1)
    ###############################
    for i in tqdm(range(divide - 1), desc="tqdm fit", mininterval=0.01):
        dfi = df.iloc[np.r_[period_unit*i:period_unit*(i+1)]]
        parameters = [dfi['close'].tolist(), dfi['volume'].tolist()]

        minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
        scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
        real_trend = dfi.close.values.tolist()
        initial_money = np.max(parameters[0])

        print('\n========================  %s start ==========================' % i)
        print('dfi %s' % str(len(dfi)))
        if i == 0:
            agent = Agent(model=model,
                          timeseries=scaled_parameters,
                          skip=skip,
                          initial_money=initial_money,
                          real_trend=real_trend,
                          real_df=dfi,
                          minmax=minmax)
        else:
            agent.change_data(timeseries=scaled_parameters,
                              skip=skip,
                              initial_money=initial_money,
                              real_trend=real_trend,
                              real_df=dfi,
                              minmax=minmax)

        agent.fit(iterations=iter, checkpoint=check)


    ###############################
    # save model
    ###############################
    import copy
    import pickle

    copy_model = copy.deepcopy(agent.model)

    if use_file_model:
        with open('app_trade_rrs_binance_trade.pkl', 'wb') as fopen:
            pickle.dump(copy_model, fopen)
            print('================== app_trade_rrs_binance_trade.pkl saved ===================')


    ###############################
    # agent buy
    ###############################
    df_buy = df.iloc[np.r_[-period_unit:0]]
    print('df_buy %s' % str(len(df_buy)))
    # print(df_buy.head())

    parameters = [df_buy['close'].tolist(), df_buy['volume'].tolist()]
    scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
    real_trend = df_buy.close.values.tolist()
    initial_money = np.max(parameters[0])

    agent.change_data(timeseries=scaled_parameters,
                      skip=skip,
                      initial_money=initial_money,
                      real_trend=real_trend,
                      real_df=df_buy,
                      minmax=minmax)

    print('\n==================  %s agent.buy() start ==================='% symbol)
    agent.buy()
    print('\n==================  %s agent.buy() end ===================' % symbol)

    ###############################
    # agent trade
    ###############################
    close = df_buy['close'].tolist()
    volume = df_buy['volume'].tolist()
    for i in range(len(df_buy) - 1):
        data = [close[i], volume[i]]
        print(data)
        print(agent.trade(data))
        print('trade done!')
