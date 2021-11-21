from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web

sns.set()

start = datetime(2019, 6, 10)
end = datetime(2020, 6, 9)
# end = datetime.now()
print(datetime.now())
# company = 'AAPL'
# company = 'MMM'
# company = 'GOOG'
company = '005930.KS' #삼성전자
# company = '008770.KS' #신라호텔
df = web.DataReader(company,'yahoo',start=start, end=end)
close = df.Close.values.tolist()


df.head()
# print('df.head():', df.head())

parameters = [df['Close'].tolist(), df['Volume'].tolist()]

window_size = 20
skip = 1
layer_size = 500
output_size = 3

# futures
fee_rate = 0.04
fee_rate_taker = 0.04
fee_rate_maker = 0.02
buy_sell_count_max = 5
print_flg = False

# origin
train_epoch = 100
fit_iterations = 500
fit_checkpoint = 20

# test
train_epoch = 100
fit_iterations = 100
fit_checkpoint = 20


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

inventory_size = 1
mean_inventory = 0.5
capital = 2
concat_parameters = np.concatenate([get_state(parameters, 20), [[inventory_size,
                                                                 mean_inventory,
                                                                 capital]]], axis=1)
input_size = concat_parameters.shape[1]
print('input_size:', input_size)


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

    # def train(self, epoch=100, print_every=1): # origin
    def train(self, epoch=train_epoch, print_every=1):  # for my setting
    # def train(self, epoch=train_epoch, print_every=1):  # for develope
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
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend):
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

    def trade(self, data):
        """
        you need to make sure the data is [close, volume]
        """
        scaled_data = [self._scale(d) for d in data]
        real_close = data[0]
        close = scaled_data[0]
        if len(self._queue) > window_size:
            self._queue.pop(0)
        self._queue.append(scaled_data)
        if len(self._queue) < window_size:
            return {'status': 'data not enough to trade', 'action': 'fail',
                    'balance': self._capital,
                    'timestamp': str(datetime.now())}
        state = self.get_state(window_size - 1,
                               self._inventory,
                               self._scaled_capital,
                               timeseries=np.array(self._queue).T.tolist())
        action = self.act(state)
        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            return {'status': 'buy 1 unit, cost %f' % (real_close),
                    'action': 'buy',
                    'balance': self._capital,
                    'timestamp': str(datetime.now())}
        elif action == 2 and len(self._inventory):
            bought_price = self._inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            scaled_bought_price = self._reverse_scale(bought_price)
            try:
                invest = ((real_close - scaled_bought_price) / scaled_bought_price) * 100
            except:
                invest = 0
            return {'status': 'sell 1 unit, price %f' % (real_close),
                    'investment': '%f %%' % (invest),
                    'balance': self._capital,
                    'action': 'sell',
                    'timestamp': str(datetime.now())}
        else:
            return {'status': 'do nothing', 'action': 'nothing',
                    'balance': self._capital,
                    'timestamp': str(datetime.now())}

    def change_data(self, timeseries, skip, initial_money, real_trend):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

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

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        # futures
        trades = []
        positions = []
        orders = []
        assets = []
        fee_total = 0
        realized_pnl_total = 0

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)

            if print_flg:
                # print('no.%s : action: %s: self.trend[t]:%s, inventory:%s' % (t, action, self.trend[t], inventory))
                print('no.%s : action: %s: self.trend[t]:%s,' % (t, action, self.trend[t]))

            if action == 1 or action == 2:
                if action == 1 and starting_money >= self.trend[t]:
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]

                elif action == 2 and len(inventory):
                    bought_price = inventory.pop(0)
                    starting_money += self.trend[t]

                # futures
                time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                symbol = 'BTCUSDT'
                price = self.trend[t]
                buy_sell_direction = 1 if action == 1 else -1
                quantity = 1
                amount = buy_sell_direction * quantity

                last_size = positions[-1][2] if positions else 0
                last_entry_price = positions[-1][3] if positions else 0
                last_realized_pnl_total = assets[-1][7] if assets else 0

                if abs(last_size) == buy_sell_count_max and last_size * buy_sell_direction > 0:
                    amount = 0

                fee = -abs(amount * fee_rate * price * 1 / 100)

                order_price = price
                buy_sell = 'buy' if action == 1 else 'sell'
                # order
                order = [time, symbol, 'market', buy_sell, order_price, amount]
                orders.append(order)


                size = last_size + amount
                entry_price = abs((last_entry_price * last_size + order_price * amount) / size) if size else 0
                mark_price = self.trend[t]

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
                    print('day %d: %s 1 unit at price %f, \n  order  %s, \n  trade  %s, \n  posi   %s, \n  asset  %s\n'
                          % (t, buy_sell, price, orders[-1], trades[-1], positions[-1], asset))

            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)
        # return ((wallet_balance - initial_money) / initial_money) * 100
        return ((starting_money - initial_money) / initial_money) * 100

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
            action = self.act(state)

            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f' % (t, self.real_trend[t],
                                                                            real_starting_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                real_bought_price = real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                states_sell.append(t)
                try:
                    invest = ((self.real_trend[t] - real_bought_price) / real_bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.real_trend[t], invest, real_starting_money)
                )
            state = self.get_state(t + 1, inventory, starting_money, self.timeseries)

        invest = ((real_starting_money - real_initial_money) / real_initial_money) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest


# stocks = [i for i in os.listdir(os.getcwd()) if '.csv' in i and not 'AMD' in i]
# print('stocks:', stocks)
stocks = ['AAPL', 'GOOG']
model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)

company = 'AAPL'
df = web.DataReader(company,'yahoo',start=start, end=end)
print('df.head() AAPL:', df.head())

real_trend = df['Close'].tolist()
stock_mean = df['Close'].mean()
stock_std = df['Close'].std()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
print('scaled_parameters:', scaled_parameters)

initial_money = np.max(parameters[0]) * 2
agent = Agent(model=model,
              timeseries=scaled_parameters,
              skip=skip,
              initial_money=initial_money,
              real_trend=real_trend)
# agent.fit(iterations=40, checkpoint=10)

volume = df['Volume'].tolist()

for i in range(40):
    print(agent.trade([real_trend[i], volume[i]]))

states_buy, states_sell, total_gains, invest = agent.buy()

'''
I want to train on all stocks I downloaded except for AMD. I want to use AMD for testing.
'''

model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)
agent = None

for no, stock in enumerate(stocks):
    print('training stock %s' % (stock))

    df = web.DataReader(stock, "yahoo", start=start, end=end)
    real_trend = df['Close'].tolist()
    stock_mean = df['Close'].mean()
    stock_std = df['Close'].std()
    parameters = [df['Close'].tolist(), df['Volume'].tolist()]
    scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
    initial_money = np.max(parameters[0]) * 2

    if no == 0:
        agent = Agent(model=model,
                      timeseries=scaled_parameters,
                      skip=skip,
                      initial_money=initial_money,
                      real_trend=real_trend)
    else:
        agent.change_data(timeseries=scaled_parameters,
                          skip=skip,
                          initial_money=initial_money,
                          real_trend=real_trend)

    # agent.fit(iterations=500, checkpoint=20)  # origin
    agent.fit(iterations=fit_iterations, checkpoint=fit_checkpoint)  # my setting
    # agent.fit(iterations=50, checkpoint=10)  # develope
    print()

company = 'AMD'
df = web.DataReader(company, 'yahoo', start=start, end=end)

real_trend = df['Close'].tolist()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
scaled_parameters = MinMaxScaler(feature_range=(100, 200)).fit_transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2

agent.change_data(timeseries=scaled_parameters,
                  skip=skip,
                  initial_money=initial_money,
                  real_trend=real_trend)

states_buy, states_sell, total_gains, invest = agent.buy()

fig = plt.figure(figsize=(15, 5))
plt.plot(real_trend, color='r', lw=2.)
plt.plot(real_trend, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
plt.plot(real_trend, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
plt.title('AMD total gains %f, total investment %f%%' % (total_gains, invest))
plt.legend()
plt.show()

import pickle

with open('agent.pkl', 'wb') as fopen:
    pickle.dump(agent, fopen)
