import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
import time

window_size = 20
skip = 1
layer_size = 500
output_size = 3


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

    # def get_weights(self):
    #     return self.weights

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
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


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

    # def set_weights(self, weights):
    #     self.weights = weights


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

        self.real_inventory = []
        self.realized_pnl = []

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
            timeseries=np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)
        print(prob)
        if action == 1 and self._scaled_capital >= close and len(self._inventory) < 1:
            self.real_inventory.append(real_close)

            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            # return {
            #     'status': 'buy 1 unit, cost %f' % (real_close),
            #     'action': 'buy',
            #     'balance': self._capital,
            #     'timestamp': str(datetime.now()),
            #     'states_pnl': str(states_pnl),
            # }
        elif action == 2 and len(self._inventory):
            self.realized_pnl.append((real_close - np.mean(self.real_inventory)) if self.real_inventory else 0)
            self.real_inventory.clear()
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


            # return {
            #     'status': 'sell 1 unit, price %f' % (real_close),
            #     'investment': invest,
            #     'gain': real_close - scaled_bought_price,
            #     'balance': self._capital,
            #     'action': 'sell',
            #     'timestamp': str(datetime.now()),
            #     'states_pnl': str(states_pnl),
            #
            # }
        else:
            pass
            # return {
            #     'status': 'do nothing',
            #     'action': 'nothing',
            #     'balance': self._capital,
            #     'timestamp': str(datetime.now()),
            #     'states_pnl': str(states_pnl),
            # }

        unrealized_pnl = 0

        if self.real_inventory:
            unrealized_pnl = (real_close - np.mean(self.real_inventory)) if self.real_inventory else 0
        real_initial_money = self.initial_money

        asset_total = real_initial_money + np.sum(self.realized_pnl) + unrealized_pnl
        pnl_total = np.sum(self.realized_pnl) + unrealized_pnl
        roe_total = (pnl_total/real_initial_money)*100
        states_pnl = [real_initial_money, asset_total, pnl_total, roe_total, np.sum(self.realized_pnl), unrealized_pnl]

        return {
                'action': action,
                'timestamp': str(datetime.now()),
                'states_pnl': str(states_pnl),
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
            if action == 1 and starting_money >= self.trend[t] and len(inventory) < 1:
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

        states_pnl = []
        realized_pnl = []
        unrealized_pnl = 0

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            print(t, prob)

            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - window_size) and len(inventory) < 1:
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                print(
                    'day %d: buy 1 unit at price %f, total balance %f'
                    % (t, self.real_trend[t], real_starting_money)
                )
            elif action == 2 and len(inventory):

                realized_pnl.append(self.real_trend[t] - np.mean(real_inventory))

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
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.real_trend[t], invest, real_starting_money)
                )

            if real_inventory:
                unrealized_pnl = self.real_trend[t] - np.mean(real_inventory) if real_inventory else 0

            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )

        invest = (
                         (real_starting_money - real_initial_money) / real_initial_money
                 ) * 100
        total_gains = real_starting_money - real_initial_money

        asset_total = real_initial_money + np.sum(realized_pnl) + unrealized_pnl
        pnl_total = np.sum(realized_pnl) + unrealized_pnl
        roe_total = (pnl_total/real_initial_money)*100
        states_pnl = [real_initial_money, asset_total, pnl_total, roe_total, np.sum(realized_pnl), unrealized_pnl]

        print(
            'Total status: day %d, \n[real_initial_money, asset_total, pnl_total, roe_total, np.sum(realized_pnl), unrealized_pnl]\n%s'
            % (t, str(states_pnl))
        )
        return states_buy, states_sell, total_gains, invest, states_pnl

df = pd.read_csv('TWTR.csv')
df.head()

parameters = [df['Close'].tolist(), df['Volume'].tolist()]

## Global parameters
window_size = 20
skip = 1
layer_size = 500
output_size = 3

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
model = Model(input_size=input_size, layer_size=layer_size, output_size=output_size)

# with open('model.pkl', 'rb') as fopen:
#     model = pickle.load(fopen)

df = pd.read_csv('TWTR.csv')
real_trend = df['Close'].tolist()
parameters = [df['Close'].tolist(), df['Volume'].tolist()]
minmax = MinMaxScaler(feature_range=(100, 200)).fit(np.array(parameters).T)
scaled_parameters = minmax.transform(np.array(parameters).T).T.tolist()
initial_money = np.max(parameters[0]) * 2

agent = Agent(model=model,
              timeseries=scaled_parameters,
              skip=skip,
              initial_money=initial_money,
              real_trend=real_trend,
              minmax=minmax)
# agent.fit(iterations=500, checkpoint=10)
agent.fit(iterations=100, checkpoint=10)

with open('model.pkl', 'wb') as fopen:
    pickle.dump(agent, fopen)

_, _, total_gains, invest, states_pnl = agent.buy()
# print(total_gains, invest, states_pnl)

# trade
close = df['Close'].tolist()
volume = df['Volume'].tolist()

for i in range(200):
    requested = agent.trade([close[i], volume[i]])
    print(str(i) +' : ' + str(requested))
