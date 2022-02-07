'''
Copyright (C) 2017-2021  Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from decimal import Decimal

from cryptofeed import FeedHandler
from cryptofeed.backends.aggregate import OHLCV
from cryptofeed.defines import BYBIT, ORDER_INFO, FILLS, CANDLES, BID, ASK, BLOCKCHAIN, FUNDING, GEMINI, L2_BOOK, L3_BOOK, LIQUIDATIONS, OPEN_INTEREST, PERPETUAL, TICKER, TRADES, INDEX
from cryptofeed.exchanges import (FTX, Binance, BinanceUS, BinanceFutures, Bitfinex, Bitflyer, AscendEX, Bitmex, Bitstamp, Bittrex, Coinbase, Gateio,
                                  HitBTC, Huobi, HuobiDM, HuobiSwap, Kraken, OKCoin, OKEx, Poloniex, Bybit, KuCoin, Bequant, Upbit, Probit)

from cryptofeed.symbols import Symbol
from cryptofeed.exchanges.phemex import Phemex
# from cryptofeed.backends.postgres import CandlesPostgres, IndexPostgres, TickerPostgres, TradePostgres, OpenInterestPostgres, LiquidationsPostgres, FundingPostgres, BookPostgres
from cryptofeed.backends.postgresn import CandlesPostgres, IndexPostgres, TickerPostgres, TradePostgres, OpenInterestPostgres, LiquidationsPostgres, FundingPostgres, BookPostgres

from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, plot_pattern_m
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from datetime import date, timedelta
import time
import json
from tapy import Indicators
import os
from pytz import UTC

try:
    from datapostgre import DataManagerfeeder
except:
    from .datapostgre import DataManagerfeeder


with open('config.json', 'r') as f:
    config = json.load(f)

####################################################################################
####################################################################################
############################# start
import ccxt
from datetime import datetime
from time import sleep
import json
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
from operator import itemgetter
from prettyprinter import pprint
import bybitwrapper
import random

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
        "leverage": 2,
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

###########    ordersize
ordersize = []
ordersize_item_d = {}
for s in both_duplicated_symbols:
    for symbol_d in bybit_symbols_result:
        if symbol_d['symbol'] == s:
            last_price = int(float(symbol_d['last_price']))
            nmlen = len(str(last_price))
            nmunit = float((1/(10**nmlen))*100)
            ordersize_item_d[s.replace('USDT', '')] = nmunit
ordersize.append(ordersize_item_d)
print('ordersize:%s' % ordersize)




def set_leverage(symbol):
    for coin in coins:
        if coin['symbol'] == symbol:
            set = client.LinearPositions.LinearPositions_saveLeverage(symbol=symbol+"USDT", buy_leverage=coin['leverage'], sell_leverage=coin['leverage']).result()
        else:
            pass

############################# end
####################################################################################
####################################################################################

from pybit import HTTP, WebSocket
import pprint
session = HTTP(
    endpoint='https://api.bybit.com',
    api_key='o2aZhUESAachytlOy5',
    api_secret='AZPK3dhKdNsRhHX2s80KxCaEsFzlt5cCuQdK',
    spot=False
)
ws = WebSocket(
    endpoint='wss://stream.bybit.com/realtime',
    subscriptions=['order', 'position'],
    api_key='o2aZhUESAachytlOy5',
    api_secret='AZPK3dhKdNsRhHX2s80KxCaEsFzlt5cCuQdK'
)

def get_status_and_candle_trajectory(symbol, df_candle, nowpoint, wtime, low, w4, high, position, status):
    try:
        status = status
        if position == 0 and status >= 0 and str(wtime) < str(nowpoint):
            c1 = df_candle.index > str(wtime)
            c2 = df_candle.index <= str(nowpoint)
            df_candle_active = df_candle[c1 & c2]
            if not df_candle_active.empty:
                maxs = np.max(df_candle_active.High.tolist())
                mins = np.min(df_candle_active.Low.tolist())

                if maxs > high + abs(high - low):
                    status += -40
                if mins < low:
                    status += -50

                if maxs <= high and mins >= w4:
                    status += 10
                    if status == 10:
                        print('%s trade condi_order if maxs <= high and mins >= w4 get_status:%s' % (symbol, 10))

                if mins < w4:
                    status += 20
                    if status == 20:
                        print('%s trade active market order if mins < w4: get_status:%s' % (symbol, 20))

            return status
        elif position == 1 and status >= 0 and str(wtime) < str(nowpoint):
            pass
    except Exception as e:
        print('get_status_and_candle_trajectory e:%s' % e)
        return -99
    return status


def agent_filter_and_update_wave_df(df_wave, symbol,  bin_size, bin_time, nowpoint, df_candle):
    c1 = df_wave['symbol'] == symbol
    c2 = df_wave['bin_size'] == bin_size
    c3 = df_wave['bin_time'] == bin_time
    c4 = df_wave['end_time'] < nowpoint
    c5 = df_wave['position'] == 0
    c6 = df_wave['status'] == 0
    df_wave = df_wave[c1 & c2 & c3 & c4
                      & c5 & c6]

    df_candle = df_candle[df_candle['Date'] <= nowpoint]
    df_candle = df_candle.drop_duplicates(['Date'], keep='first')
    df_candle = df_candle.set_index(['Date'])

    if not df_wave.empty:
        df_wave.loc[:, ('status')] = df_wave.apply(lambda x: get_status_and_candle_trajectory(symbol,
            df_candle, nowpoint, x['end_time'], x['low'], x['w4'], x['high'], x['position'], x['status']), axis=1)
    return df_wave

################
# init
################
futures = config['default']['futures']
rule = config['default']['rule']
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
window = config['default']['window']

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fractal_count = config['default']['fractal_count']
loop_count = config['default']['loop_count']

exchange = config['default']['exchange']
timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
long = config['default']['long']
o_fibo = config['default']['o_fibo']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']
symbol_random = config['default']['symbol_random']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']

plotview = config['default']['plotview']
printout = config['default']['printout']
symbols = config['symbols']['binance_futures']


def wave_manager(exchange, symbols, bin_size, bin_time, type, startpoint=None, nowpoint=None,  window=720, status_init=False):
    try:
        dm = DataManagerfeeder()
        conn = dm.get_conn()
        if status_init:
            dm.update_wave_init(conn)

        df_wave = dm.load_df_wave_startpoint_endpoint_window(conn, bin_size, bin_time, type,
                                                             startpoint=startpoint, endpoint=nowpoint, window=window)

    except Exception as e:
        print('load_df_wave_startpoint_endpoint_window in wave_manager, e:%s' % e)
        if conn:
            conn.commit()
            conn.close()
        return None

    wave_standby_l = list()
    if not df_wave.empty:
        for symbol in symbols:
            try:
                df_candle = dm.load_df_ohlc_startpoint_endpoint_window(conn, exchange, symbol, bin_size, bin_time,
                                                                       startpoint=startpoint, endpoint=nowpoint, window=window)
            except Exception as e:
                print('df_candle load_df_ohlc_startpoint_endpoint_window in wave_manager load_df:%s' % e)
                continue
            standby_df = agent_filter_and_update_wave_df(df_wave, symbol,  bin_size, bin_time, nowpoint, df_candle)
            if not standby_df.empty:
                wave_standby_l.append(standby_df)
                dm.update_wave(standby_df, -2, conn)  # column index: -2 -> status, column index: -3 -> position
    else:
        # print('wave_manager df_wave empty')
        pass

    if conn:
        conn.commit()
        conn.close()
    return wave_standby_l

pairs = Bybit.symbols()
find_USDT = 'USDT-PERP'
index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]

symbols_usdt = []
for s in index_USDTs:
    symbols_usdt.append(pairs[s])
symbols = symbols_usdt

async def ohlcv(data):
    start = time.perf_counter()
    # try:


    pairs = Bybit.symbols()
    find_USDT = 'USDT-PERP'
    index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]

    symbols_usdt = []
    for s in index_USDTs:
        symbols_usdt.append(pairs[s])
    symbols = symbols_usdt
    nowpoint = datetime.now()

    # window = 1440 / 2
    window = fractal_count * 10 if fractal_count * 10 > 100 else 100
    startpoint = None
    bin_size = 1
    bin_time = 'm'
    type = 'impulse'
    status_init = False
    dm = DataManagerfeeder()
    wave_standby_l = wave_manager(exchange, symbols, bin_size, bin_time, type, startpoint=startpoint, nowpoint=nowpoint, window=window, status_init=status_init)
    if wave_standby_l:
        conn = dm.get_conn()
        for df in wave_standby_l:
            w = df.values.tolist()
            waveid = w[-1][0]
            symbol = w[-1][1].replace('-USDT-PERP', 'USDT')
            symbolunit = w[-1][1].replace('-USDT-PERP', '')
            status = w[-1][-2]
            end_price = w[-1][-5]
            w4_price = w[-1][-6]
            start_price = w[-1][-10]
            if w[-1][-2] >= 10:  # when status in [10, 20] --> status 10 --> market,  status 20--> condi_order
                if status == 10:
                    print('%s home trade condi_order :%s' % (symbol, 10))
                    rt = session.place_conditional_order(
                        symbol=symbol,
                        order_type="Limit",
                        side="Buy",
                        qty=ordersize[0][symbolunit],
                        price=w4_price,
                        base_price=w4_price,
                        stop_px=end_price,
                        take_profit=end_price,
                        stop_loss=start_price,
                        time_in_force="GoodTillCancel",
                        order_link_id=waveid,
                        reduce_only=False,
                        trigger_by='LastPrice',
                        post_only=True,
                        close_on_trigger=False
                    )
                    print('rt_10:%s' % rt)
                    # print(client.LinearPositions.LinearPositions_tradingStop(symbol="BTCUSDT", side="Buy", take_profit=10).result())

                elif status == 20:
                    print('%s home trade active_market_order :%s' % (symbol, 20))
                    rt = session.place_active_order(
                        symbol=symbol,
                        side="Buy",
                        order_type="LIMIT",
                        qty=ordersize[0][symbolunit],
                        price=w4_price,
                        take_profit=end_price,
                        stop_loss=start_price,
                        time_in_force="GoodTillCancel",
                        reduce_only=False,
                        trigger_by='LastPrice',
                        close_on_trigger=False
                    )
                    print('rt_20:%s' % rt)
                else:
                    rt = False
                if rt and rt['ret_msg'] == 'OK':
                    dm.update_wave_key_value(int(w[0][0]), 'position', 1, conn)
        if conn:
            conn.close()
    print(f'Finished wave_manager in {round(time.perf_counter() - start, 2)} second(s)')



    # except Exception as e:
    #     print('runtrade run e:%s' % e)
    #     if conn:
    #         conn.close()
    #     pass


def main():
    config = {'log': {'filename': 'demo.log', 'level': 'DEBUG', 'disabled': False}}
    f = FeedHandler(config=config)
    f.add_feed(Coinbase(symbols=['BTC-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=1)}))
    f.run()


if __name__ == '__main__':
    if settings['check_leverage'].lower() == 'true':
        for coin in coins:
            print("Setting Leverage for ", coin['symbol'] + 'USDT', " before Starting Bot")
            set_leverage(coin['symbol'])
    main()
