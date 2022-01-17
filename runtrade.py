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

with open('config.json', 'r') as f:
    config = json.load(f)


def get_status_and_candle_trajectory(df_candle, nowpoint, wtime, w4, low, high, position, status):
    try:
        status = status
        if position == 0 and status >= 0 and str(wtime) < str(nowpoint):
            c1 = df_candle.index > str(wtime)
            c2 = df_candle.index <= str(nowpoint)
            df_candle = df_candle[c1 & c2]
            if not df_candle.empty:
                max = np.max(df_candle.High.tolist())
                min = np.min(df_candle.Low.tolist())
                status = 0
                if max >= high:
                    status += -1
                if min <= low:
                    status += -2
                if min <= w4:
                    status += 13
                    if status == 13:
                        print('trade if min <= w4: get_status:%s' % 13)
                if max < high and min > w4:
                    status += 23
                    if status == 23:
                        print('trade if max < high and min > w4 get_status:%s' % 23)
            return status
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
        df_wave.loc[:, ('status')] = df_wave.apply(lambda x: get_status_and_candle_trajectory(
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

exchange = config['default']['exchange']
timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
long = config['default']['long']
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


def wave_manager(dm, exchange, symbols, bin_size, bin_time, type, startpoint=None, nowpoint=None,  window=720, status_init=False):
    try:
        conn = dm.get_conn()
        if status_init:
            dm.update_wave_init(conn)

        df_wave = dm.load_df_wave_startpoint_endpoint_window(conn, bin_size, bin_time, type,
                                                             startpoint=startpoint, endpoint=nowpoint, window=window)

    except Exception as e:
        print('load_df_wave_startpoint_endpoint_window in wave_manager, e:%s' % e)
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
                dm.update_wave(standby_df, -2, conn)  # column index: -2 -> status, column index-3 -> position
    else:
        # print('wave_manager df_wave empty')
        pass
    conn.commit()
    conn.close()
    return wave_standby_l


async def ohlcv(data):
    start = time.perf_counter()
    try:
        from datapostgre import DataManagerfeeder
    except:
        from .datapostgre import DataManagerfeeder

    try:
        pairs = Bybit.symbols()
        find_USDT = 'USDT-PERP'
        index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]

        symbols_usdt = []
        for s in index_USDTs:
            symbols_usdt.append(pairs[s])
        symbols = symbols_usdt
        dm = DataManagerfeeder()
        nowpoint = datetime.now()

        window = 1440 / 2
        startpoint = None
        bin_size = 1
        bin_time = 'm'
        type = 'impulse'
        status_init = False
        wave_standby_l = wave_manager(dm, exchange, symbols, bin_size, bin_time, type, startpoint=startpoint, nowpoint=nowpoint, window=window, status_init=status_init)
        if wave_standby_l:
            conn = dm.get_conn()
            for df in wave_standby_l:
                w = df.values.tolist()
                if w[-1][-2] > 0:
                    dm.update_wave_key_value(int(w[0][0]), 'position', 1, conn)
            conn.close()
        print(f'Finished wave_manager in {round(time.perf_counter() - start, 2)} second(s)')
    except Exception as e:
        print('runtrade run e:%s' % e)
        conn.close()
        pass


def main():
    config = {'log': {'filename': 'demo.log', 'level': 'DEBUG', 'disabled': False}}
    f = FeedHandler(config=config)
    f.add_feed(Coinbase(symbols=['BTC-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=1)}))
    f.run()


if __name__ == '__main__':
    main()
