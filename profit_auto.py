import ccxt
import requests
from datetime import datetime
from time import sleep
import time
import json
import logging
import random
from prettyprinter import pprint
import bybitwrapper
from operator import itemgetter

with open('settings.json', 'r') as fp:
    settings = json.load(fp)
fp.close()
with open('coins.json', 'r') as fp:
    # coins = json.load(fp)
    pass
fp.close()

exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
binance = exchange_class({
    'apiKey': None,
    'secret': None,
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})
binance.load_markets()

from binance.client import Client

# basic
# client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq",
#                 "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
# futures
# client = Client("FwqJ8hQV79mUbfsUe4BuSuU9G3bmzq5lL3XaKDdRqvI8fYbPOQMU2UGJv4oM1KC0",
#                 "8yLlpdL0gvNsxbOkhQ1ZEywMY01Sr6y97YJt2vvWPwg4HL9U2MhoHdAut3btA3Wc")
binance_client = Client()

exchange_info = binance_client.get_exchange_info()
binance_symbols = []
for s in exchange_info['symbols']:
    symbol = s['symbol']
    binance_symbols.append(symbol)
# print(symbols)

find_USDT = 'USDT'
binance_index_USDTs = [i for i in range(len(binance_symbols)) if find_USDT in binance_symbols[i]]
blacklist = ['FTTUSDT']
binance_index_USDTs = [x for x in binance_index_USDTs if x not in blacklist]

binance_symbols_usdt = []
for s in binance_index_USDTs:
    binance_symbols_usdt.append(binance_symbols[s])
# print(binance_symbols_usdt)


client = bybitwrapper.bybit(test=False, api_key=settings['key'], api_secret=settings['secret'])

bybit_symbols_info = client.Market.Market_symbolInfo().result()
bybit_symbols_result = bybit_symbols_info[0]['result']

bybit_symbols_all = list(map(itemgetter('symbol'), bybit_symbols_result))
find_USDT = 'USDT'
bybit_index_USDTs = [i for i in range(len(bybit_symbols_all)) if find_USDT in bybit_symbols_all[i]]

bybit_symbols_usdt = []
for s in bybit_index_USDTs:
    bybit_symbols_usdt.append(bybit_symbols_all[s])
# print(bybit_symbols_usdt)

both_duplicated_symbols = [x for x in binance_symbols_usdt if x in bybit_symbols_usdt]

######################### symbol limit condition start #########################
random_symbol = False
symbol_length = False
print(len(both_duplicated_symbols), both_duplicated_symbols)
if random_symbol:
    both_duplicated_symbols = random.sample(both_duplicated_symbols, len(both_duplicated_symbols))
if symbol_length:
    both_duplicated_symbols = both_duplicated_symbols[:symbol_length]
######################### symbol limit condition end #########################


###########    coins
coins = list()
for s in both_duplicated_symbols:
    coin = {
        "symbol": s.replace('USDT', ''),
        "leverage": 3,
        "take_profit_percent": 0.66,
        "stop_loss_percent": 20,
        "order_size_percent_balance": 0.1,
        "long_vwap_offset": 0.1,
        "short_vwap_offset": 0.1,
        "dca_max_buy_level_1": 10,
        "dca_max_buy_level_2": 20,
        "dca_max_buy_level_3": 40,
        "dca_max_buy_level_4": 30,
        "dca_drawdown_percent_1": 2,
        "dca_drawdown_percent_2": 4,
        "dca_drawdown_percent_3": 6,
        "dca_drawdown_percent_4": 8,
        "dca_size_multiplier_1": 1.25,
        "dca_size_multiplier_2": 1.5,
        "dca_size_multiplier_3": 1.75,
        "dca_size_multiplier_4": 2,
        "lick_value": 100
    }
    coins.append(coin)


def load_jsons():
    # print("Checking Settings")
    with open('coins.json', 'r') as fp:
        # coins = json.load(fp)
        coins
        # print(coins)
    fp.close()
    with open('settings.json', 'r') as fp:
        settings = json.load(fp)
    fp.close()


def load_symbols(coins):
    symbols = []
    for coin in coins:
        symbols.append(coin['symbol'])
    return symbols


def check_positions(symbol):
    positions = client.LinearPositions.LinearPositions_myPosition(symbol=symbol + "USDT").result()
    if positions[0]['ret_msg'] == 'OK':
        for position in positions[0]['result']:
            if position['entry_price'] > 0:
                print("Position found for ", symbol, " entry price of ", position['entry_price'])
                return position
            else:
                pass

    else:
        print("API NOT RESPONSIVE AT CHECK ORDER")
        sleep(5)


def get_price_precision(symbol):
    precision = client.Symbol.Symbol_get().result()
    pprecsion = precision[0]["result"]

    for x in range(len(pprecsion)):
        if pprecsion[x]["name"] == symbol + "USDT":
            numbers = pprecsion[x]["price_filter"]["tick_size"]
            return len(numbers) - 2
    return None


def tp_calc(symbol, side):
    entry_price_data = client.LinearPositions.LinearPositions_myPosition(symbol=symbol + 'USDT').result()
    for coin in coins:
        if coin['symbol'] == symbol:
            precision = get_price_precision(symbol)
            if side == 'Buy':
                entry_price = float(entry_price_data[0]["result"][0]["entry_price"])
                price = round(entry_price + (entry_price * (coin['take_profit_percent'] / 100)), precision)
                side = 'Sell'
                return price, side
            else:
                side = 'Buy'
                entry_price = float(entry_price_data[0]["result"][1]["entry_price"])
                price = round(((entry_price * (coin['take_profit_percent'] / 100) - entry_price) * -1), precision)
                return price, side
        else:
            pass


def fetch_ticker(symbol):
    tickerDump = binance.fetch_ticker(symbol + '/USDT')
    ticker = float(tickerDump['last'])
    return ticker


def fetch_price(symbol, side):
    ticker = fetch_ticker(symbol)
    for coin in coins:
        if coin['symbol'] == symbol:
            if side == 'Buy':
                price = round(ticker + (ticker * (coin['take_profit_percent'] / 100)), 3)
                side = 'Sell'
                return price, side
            else:
                side = 'Buy'
                price = round(((ticker * (coin['take_profit_percent'] / 100) - ticker) * -1), 3)
                return price, side
        else:
            pass


def fetch_stop_price(symbol, side):
    ticker = fetch_ticker(symbol)
    for coin in coins:
        if coin['symbol'] == symbol:
            if side == 'Buy':
                price = round(ticker - (ticker * (coin['stop_loss_percent'] / 100)), 3)
                side = 'Sell'
                return price, side, price
            else:
                side = 'Buy'
                price = round(ticker + (ticker * (coin['stop_loss_percent'] / 100)), 3)
                return price, side, ticker
        else:
            pass


def cancel_orders(symbol, size, side):
    orders = client.LinearOrder.LinearOrder_getOrders(symbol=symbol + "USDT", limit='5').result()
    try:
        for order in orders[0]['result']['data']:
            if order['order_status'] != 'Filled' and order['order_status'] != 'Cancelled':
                prices = fetch_price(symbol, side)
                if size != order['qty']:
                    # print("Canceling Open Orders ", symbol)
                    cancel = client.LinearOrder.LinearOrder_cancel(symbol=symbol + "USDT",
                                                                   order_id=order['order_id']).result()
                    sleep(0.25)
                else:
                    pass
                    # print("No Changes needed for ", symbol, " Take Profit")
            else:
                pass

    except TypeError:
        pass


def cancel_stops(symbol, size, side):
    orders = client.LinearConditional.LinearConditional_getOrders(symbol=symbol + "USDT", limit='5').result()
    try:
        for order in orders[0]['result']['data']:
            # pprint(order)
            if order['order_status'] != 'Deactivated':
                # print("Canceling Open Stop Orders ", symbol)
                cancel = client.LinearConditional.LinearConditional_cancel(symbol=symbol + "USDT", stop_order_id=order[
                    'stop_order_id']).result()
                # pprint(cancel)
            else:
                pass

    except TypeError:
        pass


def set_tp(symbol, size, side):
    prices = tp_calc(symbol, side)
    cancel = client.LinearOrder.LinearOrder_cancel(symbol=symbol + "USDT").result()
    order = client.LinearOrder.LinearOrder_new(side=prices[1], symbol=symbol + "USDT", order_type="Limit", qty=size,
                                               price=prices[0], time_in_force="GoodTillCancel",
                                               reduce_only=True, close_on_trigger=False).result()


def set_sl(symbol, size, side):
    prices = fetch_stop_price(symbol, side)
    orders = client.LinearConditional.LinearConditional_getOrders(symbol=symbol + "USDT", limit='5').result()
    cancel_stops(symbol, size, side)
    # print("Setting Stop Loss ", symbol)
    order = client.LinearConditional.LinearConditional_new(order_type="Limit", side=prices[1], symbol=symbol + "USDT",
                                                           qty=size, price=prices[0],
                                                           base_price=prices[2], stop_px=prices[0],
                                                           time_in_force="GoodTillCancel",
                                                           reduce_only=False, trigger_by='LastPrice',
                                                           close_on_trigger=False).result()

    # pprint(order)


def fetch_positions():
    for coin in coins:
        symbol = coin['symbol']

        position = check_positions(symbol)

        if position != None:
            cancel_orders(symbol, position['size'], position['side'])
            set_tp(symbol, position['size'], position['side'])
            set_sl(symbol, position['size'], position['side'])
        else:
            cancel_stops(symbol, 1, 'Buy')


load_jsons()

print("Starting Take Profit & Order Manager")
while True:
    try:
        print("Checking for Positions.........")
        fetch_positions()
        sleep(settings['cooldown'])
    except Exception as e:
        print('main while e:%s' % e)
        pass
print('exit ????')