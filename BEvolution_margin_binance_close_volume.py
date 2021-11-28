import numpy as np
import pandas as pd
import time
import datetime as dt
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, normalize, minmax_scale
import pandas_datareader.data as web
from binance.client import Client
# matplotlib.use('TkAgg')
sns.set()

client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq", "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
client = Client()


def get_historical_ohlc_data(symbol, past_days=None, interval=None):
    """Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data"""

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

# 1day:1440m
past_days = 2
interval = 5  # 1, 5, 15
timeunit = 'm'
interval_str = str(interval) + timeunit
stick_cnt_per_day = past_days * 1440 / interval if timeunit == 'm' else 0
divide = 5
period_unit = math.floor(stick_cnt_per_day/divide)
print('past_days:%s interval_str:%s period_unit:%s (total_unit:%s)' % (past_days, interval_str, period_unit, stick_cnt_per_day))

window_size = 20
skip = 1
layer_size = 500
output_size = 3

# futures
fee_rate = 0.04
max_count = 5

# fit prod
# iter = 500
# check = 20

# # fit dev
# iter = 100
# check = 20
#
# fit debug
iter = 20
check = 10
fit_i_c = 'fit_i_c:%s_%s' % (iter, check)
print(fit_i_c)


inventory_size = 1
mean_inventory = 0.5
capital = 2
leverage = 1

load_model = True
# load_model = False

def get_state(parameters, t, window_size=20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = parameter[d: t + 1] if d >= 0 else -d * [parameter[0]] + parameter[0: t + 1]
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
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class Agent:
    def __init__(
        self,
        population_size,
        sigma,
        learning_rate,
        model,
        max_buy,
        max_sell,
        skip,
        window_size,
        timeseries,
        initial_money,
        real_trend
    ):
        self.window_size = window_size
        self.skip = skip
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate

        self.model = model
        self.max_buy = max_buy
        self.max_sell = max_sell

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

        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._min = np.min(self.real_trend)
        self._max = np.max(self.real_trend)
        self._scaled_capital = self._scale(self._capital)

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self._scale(self._capital)
        self._queue = []
        self._inventory = []

    def _scale(self, data):
        std = (data - self._min) / (self._max - self._min)
        return std * 100 + 100

    def _reverse_scale(self, data):
        std = (data - 100) / 100
        return (std * (self._max - self._min)) + self._min

    def get_reward(self, weights):
        self.model.weights = weights
        return self.margin('get_reward')

    def change_data(self, timeseries, skip, initial_money, real_trend):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self._initiate()

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        try:
            buy_0 = int(buy[0])
        except:
            buy_0 = 1
            # print('act error int(buy[0])')
        return np.argmax(decision[0]), buy_0

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate([state, [[len_inventory,
                                                     z_inventory,
                                                     z_capital]]], axis=1)
        return concat_parameters

    def margin(self, flg):
        initial_money = self.initial_money
        starting_money = initial_money
        quantity = 0
        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        inventory = []
        real_inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

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
            print('buy self.trend')
            print(self.trend)
        # print_flg = True
        buy_sell_count_max = max_count
        buys = [-100, 100]

        price_ex = 0
        dif_price = 0
        dif_percent = 0
        states_buy = []
        states_sell = []
        net_pnl_t = 0
        wallet_balance = 0
        wallet_balance_t = []
        net_pnl_t_s = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action, buy = self.act(state)
            buys.append(buy)

            # act
            act = [t, action, buy]
            price = self.trend[t]
            if t > 0:
                price_ex = self.trend[t - 1]
                dif_price = price - price_ex
                dif_percent = round(dif_price/100, 4)

            # chart
            chart = [price, price_ex, '||', dif_price, dif_percent]

            if print_flg:
                print(act)

            if action == 1 or action == 2:
                last_avg_price = positions[-1][0] if positions else 0
                last_hold_size = positions[-1][1] if positions else 0

                di = 1 if action == 1 else -1
                buy_sell = 'buy' if action == 1 else 'sell'

                buy_norm_minmax = minmax_scale(buys, feature_range=(0, 1), axis=0, copy=True)
                quantity = round(buy_norm_minmax[-1] * buy_sell_count_max, 4)
                quantity_limit = quantity
                if (abs(last_hold_size + di * quantity)) > buy_sell_count_max:
                    quantity_limit = buy_sell_count_max - abs(last_hold_size)

                # futures
                # time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                # symbol = 'BTCUSDT'
                transac_size = di * quantity_limit
                fee = -abs(transac_size * fee_rate * price * 1 / 100)

                # order
                order = [buy_sell, price, transac_size]
                # order = [time, symbol, 'market', buy_sell, price, transac_size]
                # orders.append(order)

                hold_size = last_hold_size + transac_size

                realized_pnl = 0
                if last_hold_size * transac_size > 0:
                    avg_price = abs((last_avg_price * last_hold_size + price * transac_size) / hold_size) if hold_size else 0
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
                roe = round((pnl / abs(last_avg_price * last_hold_size)) * 100, 2) if last_avg_price * last_hold_size else 0
                realized_pnl_total += realized_pnl

                # trade
                trade = [price, transac_size, '||', buy_sell,  fee, realized_pnl]
                # trades.clear()
                # trades.append(trade)

                # position
                position = [avg_price, hold_size, '||', pnl, roe]
                positions.clear()
                positions.append(position)

                fee_total += fee
                wallet_balance = initial_money + realized_pnl_total + fee_total + pnl
                net_pnl_t = realized_pnl_total + fee_total + pnl

                # asset
                invest_money = abs(avg_price * hold_size)
                cash_money = initial_money + realized_pnl_total + fee_total - invest_money
                asset = [wallet_balance, initial_money, '|', invest_money, cash_money]
                wallet_balance_t.append(wallet_balance)

                # analysis
                analysis = [net_pnl_t, round((net_pnl_t/initial_money)*100, 4), round((net_pnl_t/max(self.trend))*100, 4) , '|', fee_total, realized_pnl_total]
                assets.append(asset)

                if action == 1 and quantity_limit > 0 and transac_size:
                    if inventory:
                        inventory.clear()
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]
                elif action == 2 and quantity_limit > 0 and transac_size:
                    if inventory:
                        inventory.clear()
                    # bought_price = inventory.pop(0
                    inventory.append(self.trend[t])
                    starting_money += self.trend[t]
                if print_flg:
                    print('- act    %s \n  chart  %s \n  order  %s \n  trade  %s \n  posi   %s \n  asset  %s\n  analy  %s\n'
                          % (act, chart, order, trade, position, asset, analysis))

                if flg == 'buy':
                    if action == 1 and quantity_limit > 0 and transac_size:
                        if inventory:
                            inventory.clear()
                        inventory.append(self.trend[t])

                        # real_inventory.append(self.real_trend[t])
                        # real_starting_money -= self.real_trend[t]

                        starting_money -= self.trend[t]
                        states_buy.append(t)
                    elif action == 2 and quantity_limit > 0 and transac_size:
                        if inventory:
                            inventory.clear()
                        inventory.append(self.trend[t])

                        # bought_price = inventory.pop(0)
                        # real_bought_price = real_inventory.pop(0)

                        starting_money += self.trend[t]
                        real_starting_money += self.real_trend[t]

                        states_sell.append(t)
            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)
            net_pnl_t_s.append(net_pnl_t)

        roe_total = (net_pnl_t/ initial_money) * 100

        if flg == 'get_reward':
            return roe_total
        else:

            print('[%s] roe_total: %f%%, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' %
                  (symbol, round(roe_total, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   len(states_buy) + len(states_sell)))

            print('plot start')

            plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(nrows=2,
                                   ncols=1,
                                   height_ratios=[3, 1]
                                   )

            # 1st
            ax0 = plt.subplot(gs[0])
            label = '[%s] roe_total: %f%%, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' % \
                    (symbol, round(roe_total, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   str(len(states_buy) + len(states_sell)))

            plt.title(label)
            plt.xlabel(interval_str + '  ' + str(self.trend[0]) + '~' + str(self.trend[-1]))
            plt.ylabel(symbol)
            ax0.plot(self.trend, label='true close', c='b')
            ax0.plot(
                self.trend, 'o', label='predict buy', markevery=states_buy, c='g'
            )
            ax0.plot(
                self.trend, 'o', label='predict sell', markevery=states_sell, c='r'
            )
            ax0.legend()



            # 2nd
            ax1 = plt.subplot(gs[1])
            ax1.plot(net_pnl_t_s, label='net_pnl_total', c='g')
            ax1.legend()

            label = 'net_pnl_total: %s%%' % \
                    (str(net_pnl_t_s[-1]))
            plt.title(label)
            plt.xlabel(fit_i_c + '  ' + interval_str + '  ' + str(self.trend[0]) + '~' + str(self.trend[-1]))
            plt.ylabel(symbol)

            plt.tight_layout()
            plt.savefig('%s+%s+%s.svg' % (symbol, interval_str, datetime.now().strftime("%m:%d:%Y_%H:%M:%S")))
            plt.show()
            print('plot end')

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        self.margin('buy')


if not load_model:
    ###################
    ## model fit     ##
    ###################
    symbols = [
                # 'BTCUSDT',
                # 'ETHUSDT',
                'ALICEUSDT',
                # 'GTCUSDT',
                # 'TLMUSDT',
                # 'EGLDUSDT',
                # 'FTMUSDT',
                # 'AXSUSDT',
                # 'NUUSDT',
                # 'LITUSDT',
               ]
    print(symbols)
    for no, symbol in enumerate(symbols):
        print('training symbol %s' % (symbol))
        dfs = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)
        print('len(dfs):%s' % len(dfs))
        dfs = dfs[['open_date_time', 'close', 'volume']]
        # dfs['open_date_time'] = dfs['open_date_time']
        dfs['close'] = dfs['close'].astype(float)
        dfs['volume'] = dfs['volume'].astype(float)
        print(dfs.head())

        parameters = [dfs['close'].tolist(), dfs['volume'].tolist()]
        real_trend = dfs['close'].tolist()
        stock_mean = dfs['close'].mean()
        stock_std = dfs['close'].std()
        scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
        initial_money = np.max(parameters[0]) * leverage

        if no == 0:
            concat_parameters = np.concatenate([get_state(parameters, 20), [[inventory_size,
                                                                             mean_inventory,
                                                                             capital]]], axis=1)
            input_size = concat_parameters.shape[1]
            print('input_size:', input_size)

            model = Model(input_size=input_size,
                          layer_size=layer_size,
                          output_size=output_size)
            agent = Agent(
                            population_size=15,
                            sigma=0.1,
                            learning_rate=0.03,
                            max_buy=5,
                            max_sell=5,
                            window_size=window_size,
                            model=model,
                            timeseries=scaled_parameters,
                            skip=skip,
                            initial_money=initial_money,
                            real_trend=real_trend
                          )
        else:
            agent.change_data(timeseries=scaled_parameters,
                              skip=skip,
                              initial_money=initial_money,
                              real_trend=real_trend,
                              )

        agent.fit(iterations=iter, checkpoint=check)

        # for i in range(divide - 1):
        #     df_ = dfs.iloc[np.r_[period_unit*i:period_unit*(i+1)]]
        #     close = df_.close.values.tolist()
        #     print('================== %s : [%s] fit start ===================' % (symbol, i))
        #     print('df_ %s' % str(len(df_)))
        #     print('df_.index[0]:%s' % df_.index[0])
        #     print('df_.index[-1]:%s' % df_.index[-1])
        #     print(df_.head())
        #
        #     parameters = [df_['close'].tolist(), df_['volume'].tolist()]
        #     real_trend = df_['close'].tolist()
        #     stock_mean = df_['close'].mean()
        #     stock_std = df_['close'].std()
        #     scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
        #     agent.fit(iterations=iter, checkpoint=check)

    ###################
    ## save model    ##
    ###################
    with open('binance_model.pkl', 'wb') as fopen:
        pickle.dump(agent, fopen)
        print('saved binance_model.pkl ')
else:
    with open('binance_model.pkl', 'rb') as fopen:
    # with open('model.pkl', 'rb') as fopen:
        model = pickle.load(fopen)
        print('loaded binance_model.pkl ')


###################
## target symbol ##
###################
symbol = 'AXSUSDT'
print('buy symbol %s' % (symbol))
dfb = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)
print('================== %s agent buy start ===================' % (symbol))
print('len(dfb):%s' % len(dfb))
dfb = dfb[['open_date_time', 'close', 'volume']]
# dfb['open_date_time'] = dfbuy['open_date_time']
dfb['close'] = dfb['close'].astype(float)
dfb['volume'] = dfb['volume'].astype(float)


# df_buy = dfb.iloc[np.r_[-period_unit:0]]
# print('df_buy %s' % str(len(df_buy)))
# print('df_buy.index[0]:%s' % df_buy.index[0])
# print('df_buy.index[-1]:%s' % df_buy.index[-1])
print(dfb.head())

parameters = [dfb['close'].tolist(), dfb['volume'].tolist()]
real_trend = dfb['close'].tolist()
scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * leverage

if load_model:
    agent = Agent(
        population_size=15,
        sigma=0.1,
        learning_rate=0.03,
        max_buy=5,
        max_sell=5,
        window_size=window_size,
        model=model,
        timeseries=scaled_parameters,
        skip=skip,
        initial_money=initial_money,
        real_trend=real_trend
    )
else:
    agent.change_data(timeseries=scaled_parameters,
                  skip=skip,
                  initial_money=initial_money,
                  real_trend=real_trend,
                  )
agent.buy()
print('==================  %s agent buy end ===================' % symbol)
