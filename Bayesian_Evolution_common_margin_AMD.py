import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.preprocessing import normalize, minmax_scale

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
df= pd.read_csv('./AMD.csv')

close = df.Close.values.tolist()
window_size = 30
skip = 5
l = len(close) - 1

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
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        state = get_state(close, 0, self.window_size + 1)
        inventory = []
        quantity = 0


        # futures
        trades = []
        positions = []
        orders = []
        assets = []
        fee_total = 0
        realized_pnl_total = 0
        print_flg = False
        # print_flg = True
        fee_rate = 0
        buy_sell_count_max = 5
        buys = [-100, 100]

        for t in range(0, l, self.skip):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, self.window_size + 1)
            buys.append(buy)

            if print_flg:
                # print('no.%s : action: %s: self.trend[t]:%s, inventory:%s' % (t, action, self.trend[t], inventory))
                print('no.%s : action: %s: self.trend[t]:%s,' % (t, action, close[t]))

            if action == 1 or action == 2:
                last_size = positions[-1][2] if positions else 0
                last_entry_price = positions[-1][3] if positions else 0
                last_realized_pnl_total = assets[-1][7] if assets else 0


                buy_sell_direction = 1 if action == 1 else -1

                buy_norm_minmax = minmax_scale(buys, feature_range=(0, 1), axis=0, copy=True)
                quantity = round(buy_norm_minmax[-1] * buy_sell_count_max, 4)

                if (abs(last_size + buy_sell_direction * quantity)) > buy_sell_count_max:
                    quantity = buy_sell_count_max - abs(last_size)

                # futures
                time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                symbol = 'BTCUSDT'
                price = close[t]
                buy_sell_direction = 1 if action == 1 else -1

                amount = buy_sell_direction * quantity

                fee = -abs(amount * fee_rate * price * 1 / 100)

                order_price = price
                buy_sell = 'buy' if action == 1 else 'sell'
                # order
                order = [time, symbol, 'market', buy_sell, order_price, amount]
                orders.append(order)

                size = last_size + amount
                entry_price = abs((last_entry_price * last_size + order_price * amount) / size) if size else 0
                mark_price = close[t]

                pnl = (price - last_entry_price) * last_size if last_entry_price else 0
                realized_pnl = (last_entry_price - price) * amount \
                    if last_entry_price != 0 and abs(last_size + amount) <= abs(last_size) else 0

                roe = round((pnl / abs(last_entry_price * last_size)) * 100, 2) if last_entry_price * last_size else 0
                realized_pnl_total += realized_pnl

                # trade
                trade = [time, symbol, buy_sell, price, amount, fee, pnl, realized_pnl]
                trades.append(trade)

                # position
                position = [time, symbol, size, entry_price, mark_price, pnl, roe]
                positions.append(position)

                # asset
                fee_total += fee
                wallet_balance = initial_money + last_realized_pnl_total + fee_total + pnl
                net_pnl_total = wallet_balance - initial_money

                asset = [time, wallet_balance, initial_money, pnl, roe, fee_total, realized_pnl_total, net_pnl_total]
                assets.append(asset)

                if print_flg:
                    print('day %d: %s %s unit at price %f, \n  order  %s, \n  trade  %s, \n  posi   %s, \n  asset  %s\n'
                          % (t, buy_sell, round(quantity, 2), price, orders[-1], trades[-1], positions[-1], asset))


            state = next_state
        return ((wallet_balance - initial_money) / initial_money) * 100
        #
        # return ((initial_money - starting_money) / starting_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def buy(self):
        initial_money = self.initial_money
        state = get_state(close, 0, self.window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0

        # futures
        trades = []
        positions = []
        orders = []
        assets = []
        fee_total = 0
        realized_pnl_total = 0
        print_flg = True
        fee_rate = 0
        buy_sell_count_max = 5
        buys = [-100, 100]
        roe = 0
        for t in range(0, l, self.skip):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, self.window_size + 1)
            buys.append(buy)

            if print_flg:
                # print('no.%s : action: %s: self.trend[t]:%s, inventory:%s' % (t, action, self.trend[t], inventory))
                print('no.%s : action: %s: self.trend[t]:%s,' % (t, action, close[t]))

            if action == 1 or action == 2:
                last_size = positions[-1][2] if positions else 0
                last_entry_price = positions[-1][3] if positions else 0
                last_realized_pnl_total = assets[-1][7] if assets else 0

                # futures
                time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                symbol = 'BTCUSDT'
                price = close[t]
                buy_sell_direction = 1 if action == 1 else -1

                buy_norm_minmax = minmax_scale(buys, feature_range=(0, 1), axis=0, copy=True)
                quantity = round(buy_norm_minmax[-1] * buy_sell_count_max, 4)

                if (abs(last_size + buy_sell_direction * quantity)) > buy_sell_count_max:
                    quantity = buy_sell_count_max - abs(last_size)

                if buy_sell_direction == 1 and quantity > 0:
                    states_buy.append(price)
                elif buy_sell_direction == -1 and quantity > 0:
                    states_sell.append(price)

                amount = buy_sell_direction * quantity

                fee = -abs(amount * fee_rate * price * 1 / 100)

                order_price = price
                buy_sell = 'buy' if action == 1 else 'sell'
                # order
                order = [time, symbol, 'market', buy_sell, order_price, amount]
                orders.append(order)

                size = last_size + amount
                entry_price = abs((last_entry_price * last_size + order_price * amount) / size) if size else 0
                mark_price = close[t]

                pnl = (price - last_entry_price) * last_size if last_entry_price else 0
                realized_pnl = (last_entry_price - price) * amount \
                    if last_entry_price != 0 and abs(last_size + amount) <= abs(last_size) else 0

                roe = round((pnl / abs(last_entry_price * last_size)) * 100, 2) if last_entry_price * last_size else 0
                realized_pnl_total += realized_pnl

                # trade
                trade = [time, symbol, buy_sell, price, amount, fee, pnl, realized_pnl]
                trades.append(trade)

                # position
                position = [time, symbol, size, entry_price, mark_price, pnl, roe]
                positions.append(position)

                # asset
                fee_total += fee
                wallet_balance = initial_money + last_realized_pnl_total + fee_total + pnl
                net_pnl_total = wallet_balance - initial_money

                asset = [time, wallet_balance, initial_money, pnl, roe, fee_total, realized_pnl_total, net_pnl_total]
                assets.append(asset)

                if print_flg:
                    print('day %d: %s %s unit at price %f, \n  order  %s, \n  trade  %s, \n  posi   %s, \n  asset  %s\n'
                          % (t, buy_sell, round(quantity, 2), price, orders[-1], trades[-1], positions[-1], asset))

            state = next_state
            roe = ((wallet_balance - initial_money) / initial_money) * 100
            if print_flg:
                print('\nroe: %s%, wallet_balance:%s, initial_money:%s' % (round(roe, 2), wallet_balance, initial_money))
        return round(roe, 2)

        plt.figure(figsize = (20, 10))
        plt.plot(close, label = 'true close', c = 'g')
        plt.plot(
            close, 'X', label = 'predict buy', markevery = states_buy, c = 'b'
        )
        plt.plot(
            close, 'o', label = 'predict sell', markevery = states_sell, c = 'r'
        )
        plt.legend()
        plt.show()

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
              money = 10000, 
              max_buy = 5, 
              max_sell = 5, 
              skip = 1, 
              window_size = 30)
agent.fit(500, 100)
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