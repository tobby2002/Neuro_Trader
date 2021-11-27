import numpy as np
import pandas as pd
import time
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math
import random
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, normalize, minmax_scale
import pandas_datareader.data as web
from binance.client import Client
# matplotlib.use('TkAgg')

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
past_days = 30

interval = 15  # 1, 5, 15
timeunit = 'm'
interval_str = str(interval) + timeunit

if timeunit == 'm':
    stick_cnt_per_day = past_days * 1440 / interval

divide = 5
period_unit = math.floor(stick_cnt_per_day/divide)

print('past_days:%s interval_str:%s period_unit:%s (total_unit:%s)' % (past_days, interval_str, period_unit, stick_cnt_per_day))
symbols = ['BTCUSDT','ALICEUSDT']
# symbols = ['BTCUSDT','ALICEUSDT','GTCUSDT','TLMUUSDT','ETHUSDT', 'EGLDUSDT']



symbol = 'ALICEUSDT'
symbol = 'FTMUSDT'
symbol = 'AXSUSDT'
symbol = 'GTCUSDT'  # --> buy error
symbol = 'TLMUSDT'  # --> buy error
symbol = 'ETHUSDT'

symbol = 'BTCUSDT'
# symbol = 'EGLDUSDT'
# symbol = 'NUUSDT'
symbol = 'MASKUSDT'
print('symbol:%s' % symbol)
# dfo = get_historical_ohlc_data(symbol, past_days = 150, interval = '5m')
dfo = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)

# dfo = dfo.set_index("open_date_time")
# dfo['close'] = dfo['close'].astype(float)
# dfo = dfo['close']


# dfo = dfo.apply(pd.to_numeric, errors='coerce')
print('len(dfo):%s' % len(dfo))
# dfo = dfo.loc[:, 'close':'volume']
dfo = dfo[['open_date_time', 'close', 'volume']]
dfo['open_date_time'] = dfo['open_date_time']
dfo['close'] = dfo['close'].astype(float)
dfo['volume'] = dfo['volume'].astype(float)
print(dfo.head())
sns.set()

def get_state(data, t, n):
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])

company = 'bitcoin'
df = dfo.iloc[np.r_[0:period_unit]]
print('df.index[0]:%s' % df.index[0])
print('df.index[-1]:%s' % df.index[-1])
# df = df.iloc[np.r_[0:2, -2:0]]

parameters = [df['close'].tolist(), df['volume'].tolist()]
close = df.close.values.tolist()

# scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2
# initial_money = 10000
print('initial_money:%s' % initial_money)


window_size = 30
skip = 5
l = len(close) - 1

# futures
fee_rate = 0.04
max_count = 5

# fit prod
iter = 500
check = 20

# # fit dev
# iter = 100
# check = 20
#
# fit debug
# iter = 40
# check = 20
fit_i_c = 'fit_i_c:%s_%s' % (iter, check)
print(fit_i_c)

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
        money,
        max_buy,
        max_sell,
        skip,
        window_size,
    ):
        self.window_size = window_size
        self.skip = skip
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.model = model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        try:
            buy_0 = int(buy[0])
        except:
            buy_0 = 0
            print('act error int(buy[0])')
        return np.argmax(decision[0]), buy_0

    def get_reward(self, weights):
        self.model.weights = weights
        return self.margin('get_reward')
        #
        # return ((initial_money - starting_money) / starting_money) * 100


    def margin(self, flg):
        initial_money = self.initial_money
        starting_money = initial_money
        state = get_state(close, 0, self.window_size + 1)
        inventory = []
        quantity = 0

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
        # print_flg = True
        # futures
        # fee_rate = 0.04
        buy_sell_count_max = max_count
        buys = [-100, 100]

        price_ex = 0
        dif_price = 0
        dif_percent = 0
        states_buy = []
        states_sell = []
        net_pnl_total = 0
        wallet_balance = 0
        wallet_balance_t = []
        net_pnl_total_t = []

        for t in range(0, l, self.skip):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, self.window_size + 1)
            buys.append(buy)

            # act
            act = [t, action, buy]
            # acts.append(act)
            price = close[t]
            if t > 0:
                price_ex = close[t - 1]
                dif_price = price - price_ex
                dif_percent = round(dif_price/100, 4)

            # chart
            chart = [price, price_ex, '||', dif_price, dif_percent]
            # charts.append(chart)

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
                net_pnl_total = realized_pnl_total + fee_total + pnl

                # asset
                invest_money = abs(avg_price * hold_size)
                cash_money = initial_money + realized_pnl_total + fee_total - invest_money
                asset = [wallet_balance, initial_money, '|', invest_money, cash_money]
                wallet_balance_t.append(wallet_balance)

                # analysis
                analysis = [net_pnl_total, round((net_pnl_total/initial_money)*100, 4), round((net_pnl_total/max(close))*100, 4) , '|', fee_total, realized_pnl_total]
                net_pnl_total_t.append(net_pnl_total)
                assets.append(asset)

                if print_flg:
                    print('- act    %s \n  chart  %s \n  order  %s \n  trade  %s \n  posi   %s \n  asset  %s\n  analy  %s\n'
                          % (act, chart, order, trade, position, asset, analysis))

                if flg == 'buy':
                    if action == 1 and quantity_limit > 0 and transac_size:
                        states_buy.append(t)
                    elif action == 2 and quantity_limit > 0 and transac_size:
                        states_sell.append(t)

            state = next_state

        roe_total = ((net_pnl_total) / initial_money) * 100

        if flg == 'get_reward':
            return roe_total
        else:

            print('[%s] roe_total: %s, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' %
                  (symbol, round(roe_total, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   len(states_buy) + len(states_sell)))

            print('plot start')



            plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(nrows=2,
                                   ncols=1,
                                   height_ratios=[3, 1]
                                   )


            # 1st
            # plt.subplot(2,1,1)
            ax0 = plt.subplot(gs[0])
            # label = '[%s] net_pnl_total: %s' % \
            #         (symbol, net_pnl_total_t[-1])
            label = '[%s] roe_total: %s, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' % \
                    (symbol, round(roe_total, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   str(len(states_buy) + len(states_sell)))

            plt.title(label)
            plt.xlabel(interval_str + '  ' + str(close[0]) + '~' + str(close[-1]))
            plt.ylabel(symbol)
            ax0.plot(close, label='true close', c='b')
            ax0.plot(
                close, 'X', label='predict buy', markevery=states_buy, c='g'
            )
            ax0.plot(
                close, 'o', label='predict sell', markevery=states_sell, c='r'
            )
            ax0.legend()



            # 2nt
            # plt.subplot(2,1,2)
            ax1 = plt.subplot(gs[1])

            ax1.plot(net_pnl_total_t, label='net_pnl_total', c='g')
            ax1.legend()

            label = '[%s] net_pnl_total: %s' % \
                    (symbol, net_pnl_total_t[-1])
            plt.title(label)
            plt.xlabel(fit_i_c + '  ' + interval_str + '  ' + str(close[0]) + '~' + str(close[-1]))
            plt.ylabel(symbol)

            plt.tight_layout()
            plt.savefig('%s+%s+%s.svg' % (symbol, interval_str, datetime.now().strftime("%m:%d:%Y_%H:%M:%S")))
            plt.show()
            print('plot end')

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        self.margin('buy')

# def best_agent(
#     window_size, skip, population_size, sigma, learning_rate, size_network):
#     model = Model(window_size, size_network, 3)
#     agent = Agent(
#         population_size,
#         sigma,
#         learning_rate,
#         model,
#         10000,
#         5,
#         5,
#         skip,
#         window_size) #comma after window size, check
#     try:
#         agent.fit(100, 1000)
#         return agent.es.reward_function(agent.es.weights)
#     except:
#         return 0
#
# def find_best_agent(
#     window_size, skip, population_size, sigma, learning_rate, size_network
# ):
#     global accbest
#     param = {
#         'window_size': int(np.around(window_size)),
#         'skip': int(np.around(skip)),
#         'population_size': int(np.around(population_size)),
#         'sigma': max(min(sigma, 1), 0.0001),
#         'learning_rate': max(min(learning_rate, 0.5), 0.000001),
#         'size_network': int(np.around(size_network)),
#     }
#     print('\nSearch parameters %s' % (param))
#     investment = best_agent(**param)
#     print('stop after 100 iteration with investment %f' % (investment))
#     if investment > accbest:
#         costbest = investment
#     return investment
#
# accbest = 0.0
# NN_BAYESIAN = BayesianOptimization(
#     find_best_agent,
#     {
#         'window_size': (2, 50),
#         'skip': (1, 15),
#         'population_size': (1, 50),
#         'sigma': (0.01, 0.99),
#         'learning_rate': (0.000001, 0.49),
#         'size_network': (10, 1000),
#     },
# )
# NN_BAYESIAN.maximize(init_points = 30, n_iter = 50, acq = 'ei', xi = 0.0)
#
# print('Best AGENT accuracy value: %f' % NN_BAYESIAN.res['max']['max_val'])
# print('Best AGENT parameters: ', NN_BAYESIAN.res['max']['max_params'])
#
#
# #Best parameters according to me
# best_agent(
#     window_size = 30,
#     skip = 1,
#     population_size = 15,
#     sigma = 0.1,
#     learning_rate = 0.03,
#     size_network = 500)
#
# #Best parameters according to bayesian optimization
#
# best_agent(
#     window_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['window_size'])),
#     skip = int(np.around(NN_BAYESIAN.res['max']['max_params']['skip'])),
#     population_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['population_size'])),
#     sigma = NN_BAYESIAN.res['max']['max_params']['sigma'],
#     learning_rate = NN_BAYESIAN.res['max']['max_params']['learning_rate'],
#     size_network = int(np.around(NN_BAYESIAN.res['max']['max_params']['size_network'])))

#Best hyperparameters according to me


model = Model(input_size=30,
              layer_size=500,
              output_size=3)

agent = Agent(population_size=15,
              sigma=0.1,
              learning_rate=0.03,
              model=model,
              money=initial_money,
              max_buy=5,
              max_sell=5,
              skip=1,
              window_size=30)


# agent.fit  (train 4 : buy 1)
for i in range(divide - 1):
    df_ = dfo.iloc[np.r_[period_unit*i:period_unit*(i+1)]]
    close = df_.close.values.tolist()
    # agent.fit(500, 100)

    print('==================  %s start ===================' % i)
    print('df_ %s' % str(len(df_)))
    print('df_.index[0]:%s' % df_.index[0])
    print('df_.index[-1]:%s' % df_.index[-1])
    print(df_.head())

    # agent.fit(500, 100)
    agent.fit(iter, check)


df_buy = dfo.iloc[np.r_[-period_unit:0]]
print('df_buy %s' % str(len(df_buy)))
print('df_buy.index[0]:%s' % df_buy.index[0])
print('df_buy.index[-1]:%s' % df_buy.index[-1])
print(df_.head())
close = df_buy.close.values.tolist()
print('==================  %s agent.buy() start ==================='% symbol)
agent.buy()
print('==================  %s agent.buy() end ===================' % symbol)

#Best hyperparamters according to bayesian optimization

# model = Model(input_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['window_size'])),
#               layer_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['size_network'])),
#               output_size = 3)
# agent = Agent(population_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['population_size'])),
#               sigma = NN_BAYESIAN.res['max']['max_params']['sigma'],
#               learning_rate = NN_BAYESIAN.res['max']['max_params']['learning_rate'],
#               model = model,
#               money = 10000,
#               max_buy = 5,
#               max_sell = 5,
#               skip = int(np.around(NN_BAYESIAN.res['max']['max_params']['skip'])),
#               window_size = int(np.around(NN_BAYESIAN.res['max']['max_params']['window_size'])))
# agent.fit(500, 100)
# agent.buy()