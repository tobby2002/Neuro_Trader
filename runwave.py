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

try:
    from datapostgre import DataManagerfeeder
except:
    from .datapostgre import DataManagerfeeder

with open('config.json', 'r') as f:
    config = json.load(f)

def check_has_same_wavepattern_df(df_wave, wavepattern):
    l = df_wave.wave.tolist()
    if l:
        for i in l:
            # eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(i[1]))
            eq_values = np.array_equal(np.array(wavepattern.values), np.array(i[2]))
            # if eq_dates and eq_values:
            if eq_values:
                return True
    return False


def get_status_and_candle_trajectory(df_candle, nowpoint, wtime, w4, low, high, position, status):
    try:
        if status == 0 and str(wtime) < str(nowpoint):
            print(wtime, nowpoint)
            max = np.max(df_candle[wtime:nowpoint].High.tolist())
            if max >= high:
                return 2
            min = np.min(df_candle[wtime:nowpoint].Low.tolist())
            if min <= w4:
                return 1
            if max < high and min > w4:
                return 0
            return -1
    except Exception as e:
        print('get_status_and_candle_trajectory:%s' % e)
        return -2

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


def fractals_low_loopA(df, fractal_count=5, loop_count=1):
    for c in range(loop_count):
        window = 2 * fractal_count + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fractal_count] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df


def fractals_low_loop(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_low']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_high_loop(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_high']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_only_hi_lo(df):
    i = Indicators(df)
    i.fractals()
    df = i.df
    df = df[~(df[['fractals_high', 'fractals_low']] == 0).all(axis=1)]
    df = df.dropna()
    df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df


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

def check_has_same_wavepattern(w_l, wavepattern):
    for i in w_l:
        eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(i[-1].dates))
        eq_values = np.array_equal(np.array(wavepattern.values), np.array(i[-1].values))
        if eq_dates and eq_values:
            return True
    return False

def wave_analyzer(exchange, symbols, bin_size, bin_time, up_to_count, startpoint, nowpoint, window, truncate=False):
    dm = DataManagerfeeder()
    conn = dm.get_conn()
    if truncate:
        dm.truncate_tablename(conn, 'wave')
        print('truncated wave')
    for symbol in symbols:
        try:
            df_candle = dm.load_df_ohlc_startpoint_endpoint_window(conn, exchange, symbol, bin_size, bin_time, startpoint, nowpoint, window)
            df_wave = dm.load_df_wave(conn)
        except Exception as e:
            print('df_candle and df_wave dm.load_df:%s' % e)
            continue

        if nowpoint:
            c = df_candle['Date'] <= nowpoint
            if startpoint:
                c = c & (df_candle['Date'] >= startpoint)
            df_candle = df_candle[c]

        if df_candle.empty:
            continue

        # if df_candle.size >= window:
        #     df_candle = df_candle.iloc[-window:]

        df_all = df_candle
        df_all.reset_index(inplace=True, drop=False)
        df_lows = fractals_low_loopA(df_all, fractal_count=fractal_count, loop_count=loop_count)
        df_lows_plot = df_lows[['Date', 'Low']]
        wa = WaveAnalyzer(df=df_all, verbose=False)
        up_to_count = up_to_count if up_to_count else 3
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)

        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        wavepattern_up_l = []
        wave_option_plot_l = []
        wavepatterns = set()

        mins = df_lows.index.tolist()
        wavepattern_l = []
        wave_option_plot_l = []

        for i in mins:
            for new_option_impulse in wave_options_impulse.options_sorted:
                waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
                if waves_up:
                    wavepattern = WavePattern(waves_up, verbose=True)
                    # new condition by neo
                    if (wavepattern.idx_end - wavepattern.idx_start + 1) >= fractal_count / 2:
                        for rule in rules_to_check:
                            if wavepattern.check_rule(rule):
                                if wavepattern in wavepatterns:
                                    continue
                                else:
                                    wavepatterns.add(wavepattern)
                                    if not check_has_same_wavepattern(wavepattern_l, wavepattern):
                                        if dm.check_no_duplicate_wave(symbol, bin_size, bin_time, wavepattern.dates[0], wavepattern.dates[9], conn):
                                            # new condition by neo
                                            if not check_has_same_wavepattern_df(df_wave, wavepattern):
                                                wavepatterns.add(wavepattern)
                                                wavepattern_l.append([i, wavepattern.dates[0], id(wavepattern), wavepattern])
                                                w_info = [
                                                    int(time.time() * 1000.0),
                                                    symbol,
                                                    bin_size,
                                                    bin_time,
                                                    rule.name,
                                                    wavepattern.dates[0],
                                                    wavepattern.dates[9],
                                                    wavepattern.values[0],
                                                    wavepattern.values[1],
                                                    wavepattern.values[3],
                                                    wavepattern.values[5],
                                                    wavepattern.values[7],
                                                    wavepattern.values[9],
                                                    'wavedata',
                                                    0,
                                                    0,
                                                    str(datetime.now())

                                                ]

                                                dm.save_wave(w_info, conn)
                                                wavepattern_up_l.append(wavepattern)
                                                wave_option_plot_l.append([
                                                    [str(wavepattern.waves['wave5'].dates[1])],
                                                    [wavepattern.waves['wave5'].high],
                                                    [str(new_option_impulse)]
                                                ])
                                                # print(f'{rule.name} found: {new_option_impulse.values}')
                                                print('%s %s:save_wave:%s ' % (symbol, str(new_option_impulse.values), str(w_info)))

        if plotview:
            if wavepattern_up_l:
                t = str(symbol + '_fractal%s_'% str(fractal_count) + str(bin_size) + 'm_' + rule.name + ' ')
                plot_pattern_m(df=df_all, wave_pattern=wavepattern_up_l,
                               df_plot=df_lows_plot,
                               wave_options=wave_option_plot_l, title=t)
                # print(t)

    if conn:
        conn.commit()
        conn.close()


##################################################
#### 1 (wave_analyzer scheduler)
##################################################
pairs = Bybit.symbols()
find_USDT = 'USDT-PERP'
index_USDTs = [i for i in range(len(pairs)) if find_USDT in pairs[i]]

symbols_usdt = []
for s in index_USDTs:
    symbols_usdt.append(pairs[s])
symbols = symbols_usdt


async def ohlcv(data):
    start = time.perf_counter()
    try:
        # window = 1440 / 2
        window = fractal_count * 10 if fractal_count * 10 > 100 else 100
        nowpoint = datetime.now()
        truncate = False
        bin_size = 1
        bin_time = 'm'
        wave_analyzer(exchange, symbols, bin_size, bin_time, up_to_count=up_to_count,
                                               startpoint=None, nowpoint=nowpoint, window=window, truncate=truncate)
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')
    except Exception as e:
        print('ohlcv:Exception e: %s' % e)
        pass


def main():
    config = {'log': {'filename': 'demo.log', 'level': 'DEBUG', 'disabled': False}}
    f = FeedHandler(config=config)
    f.add_feed(Coinbase(symbols=['BTC-USD'], channels=[TRADES], callbacks={TRADES: OHLCV(ohlcv, window=1)}))

    f.run()


if __name__ == '__main__':
    pid = os.getpid()
    print('wave_analyzer start pid: %s' % pid)
    main()
