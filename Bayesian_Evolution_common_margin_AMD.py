import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, normalize, minmax_scale
import pandas_datareader.data as web

sns.set()

import pkg_resources
import types

def get_state(data, t, n):
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])

company = '005930.KS' #삼성전자
# company = '008770.KS' #신라호텔
start = datetime(2019, 6, 10)
end = datetime(2020, 6, 9)
# df = web.DataReader(company,'yahoo',start=start, end=end)
# df = web.DataReader('AAPL','yahoo',start=start, end=end)
# df= pd.read_csv('./dataset/xyz.csv')
# df= pd.read_csv('./AMD.csv')

company = 'AMD'
df = web.DataReader(company, 'yahoo', start=start, end=end)

# real_trend = df['Close'].tolist()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2


close = df.Close.values.tolist()
window_size = 30
skip = 5
l = len(close) - 1

# futures
fee_rate = 0.04

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
        return np.argmax(decision[0]), int(buy[0])

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
        buy_sell_count_max = 5
        buys = [-100, 100]

        price_ex = 0
        dif_price = 0
        dif_percent = 0
        states_buy = []
        states_sell = []

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
            chart = [price, price_ex, dif_price, dif_percent]
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
                orders.append(order)

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
                trade = [price, transac_size, buy_sell, fee, realized_pnl]
                trades.append(trade)

                # position
                position = [avg_price, hold_size, pnl, roe]
                positions.clear()
                positions.append(position)

                fee_total += fee
                wallet_balance = initial_money + realized_pnl_total + fee_total + pnl
                net_pnl_total = realized_pnl_total + fee_total + pnl

                # asset
                asset = [wallet_balance, initial_money, fee_total, realized_pnl_total, net_pnl_total, (net_pnl_total/initial_money)*100, (net_pnl_total/max(close))*100 ]
                assets.append(asset)

                if print_flg:
                    print('- act    %s \n  chart  %s \n  order  %s \n  trade  %s \n  posi   %s \n  asset  %s\n'
                          % (act, chart, order, trade, position, asset))

                if flg == 'buy':
                    if action == 1 and quantity_limit > 0:
                        states_buy.append(t)
                    elif action == 2 and quantity_limit > 0:
                        states_sell.append(t)

            state = next_state

        if flg == 'get_reward':
            return ((net_pnl_total) / initial_money) * 100
        else:

            print('  + roe: %s, wallet_balance:%s, initial_money:%s, buy %s, sell %s, count %s' %
                  (round(roe, 2), wallet_balance, initial_money, len(states_buy), len(states_sell),
                   len(states_buy) + len(states_sell)))

            print('plot')
            plt.figure(figsize=(20, 10))
            plt.plot(close, label='true close', c='g')
            plt.plot(
                close, 'X', label='predict buy', markevery=states_buy, c='b'
            )
            plt.plot(
                close, 'o', label='predict sell', markevery=states_sell, c='r'
            )
            plt.legend()
            plt.show()

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

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


model = Model(input_size = 30, 
              layer_size = 500, 
              output_size = 3)
agent = Agent(population_size = 15, 
              sigma = 0.1, 
              learning_rate = 0.03, 
              model = model, 
              # money = 10000,
              money = initial_money,
              max_buy = 5,
              max_sell = 5, 
              skip = 1, 
              window_size = 30)
# agent.fit(500, 100)
agent.fit(100, 20)
agent.buy()


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