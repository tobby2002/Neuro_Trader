from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction, DownImpulse
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, plot_pattern_m
import pandas as pd
import numpy as np
import math
from binance.client import Client
import datetime as dt
from tapy import Indicators
import time
import multiprocessing
from functools import partial
from ta import add_all_ta_features
from ta.utils import dropna
import random
from tqdm import tqdm
from scipy.signal import argrelextrema
import shutup; shutup.please()
from datetime import datetime
from typing import Union, Optional, Dict
import dateparser
import pytz
import json
from pybit import HTTP

session = HTTP(
    endpoint='https://api.bybit.com',
    api_key='o2aZhUESAachytlOy5',
    api_secret='AZPK3dhKdNsRhHX2s80KxCaEsFzlt5cCuQdK',
    spot=False
)

with open('config.json', 'r') as f:
    config = json.load(f)

exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
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

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fcnt = config['default']['fcnt']
loop_count = config['default']['loop_count']


timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
condi_same_date = config['default']['condi_same_date']
long = config['default']['long']
o_fibo = config['default']['o_fibo']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']
symbol_random = config['default']['symbol_random']
symbol_duplicated = config['default']['symbol_duplicated']
symbol_last = config['default']['symbol_last']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']


intersect_idx = config['default']['intersect_idx']
plotview = config['default']['plotview']
printout = config['default']['printout']


def print_condition():
    print('-------------------------------')
    print('exchange:%s' % str(exchange))
    print('exchange_symbol:%s' % str(exchange_symbol))
    print('futures:%s' % str(futures))
    print('type:%s' % str(type))
    print('leverage:%s' % str(leverage))
    print('seed:%s' % str(seed))
    print('fee:%s%%' % str(fee*100))
    print('fee_maker:%s%%' % str(fee_maker*100))
    print('fee_taker:%s%%' % str(fee_taker*100))
    print('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
    if futures:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

    else:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

    print('timeframe: %s' % timeframe)
    print('period_days_ago: %s' % period_days_ago)
    print('period_days_ago_till: %s' % period_days_ago_till)
    print('period_interval: %s' % period_interval)
    print('round_trip_count: %s' % round_trip_count)
    print('compounding: %s' % compounding)
    print('fcnt: %s' % fcnt)
    print('loop_count: %s' % loop_count)

    print('symbol_duplicated: %s' % symbol_duplicated)
    print('symbol_random: %s' % symbol_random)
    print('symbol_last: %s' % symbol_last)
    print('symbol_length: %s' % symbol_length)

    # print('timeframe: %s' % timeframe)
    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    print('period: %s ~ %s' % (start_dt, end_dt))
    print('up_to_count: %s' % up_to_count)
    print('condi_same_date: %s' % condi_same_date)
    # print('long: %s' % long)
    print('o_fibo: %s' % o_fibo)
    print('h_fibo: %s' % h_fibo)
    print('l_fibo: %s' % l_fibo)

    print('intersect_idx: %s' % intersect_idx)
    print('plotview: %s' % plotview)
    print('printout: %s' % printout)
    print('-------------------------------')

client = None
if not futures:
    # basic
    client_basic = Client("basic_secret_key",
                    "basic_secret_value")
    client = client_basic
else:
    # futures
    client_futures = Client("futures_secret_key",
                    "futures_secret_value")
    client = client_futures


import threading
import functools
import time

def synchronized(wrapped):
    lock = threading.Lock()
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            # print ("Calling '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            result = wrapped(*args, **kwargs)
            # print ("Done '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            return result
    return _wrap



def date_to_milliseconds(date_str: str) -> int:
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    """
    # get epoch value in UTC
    epoch: datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d: Optional[datetime] = dateparser.parse(date_str, settings={'TIMEZONE': "UTC"})
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def get_historical_klines_pd(symbol, interval, start_date_str, end_date_str, start_int, end_int):
    """Get Historical Klines from Bybit
    See dateparse docs for valid start and end string formats
    http://dateparser.readthedocs.io/en/latest/
    If using offset strings for dates add "UTC" to date string
    e.g. "now UTC", "11 hours ago UTC"
    :param symbol: Name of symbol pair -- BTCUSD, ETCUSD, EOSUSD, XRPUSD
    :type symbol: str
    :param interval: Bybit Kline interval -- 1 3 5 15 30 60 120 240 360 720 "D" "M" "W" "Y"
    :type interval: str
    :param start_int: Start date string in UTC format
    :type start_int: str
    :param end_int: optional - end date string in UTC format
    :type end_int: str
    :return: list of OHLCV values
    """

    # set parameters for kline()
    timeframe = str(interval)
    start_ts = int(date_to_milliseconds(start_date_str)/1000)
    end_ts = None
    if end_date_str:
        end_ts = int(date_to_milliseconds(end_date_str)/1000)
    else:
        end_ts = int(date_to_milliseconds('now')/1000)

    # init our list
    output_data = []


    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    delta_seconds = (start_int - end_int) * 1440
    kline_loop_cnt = int(delta_seconds/200)
    kline_loop_last_limit = int(delta_seconds%200)

    symbol_existed = False
    # loop counter
    idx = 1
    limit = 200

    while True:
        # fetch the klines from start_ts up to max 200 entries
        temp_dict = session.query_mark_price_kline(
            symbol=symbol,
            interval=timeframe,
            limit=limit,
            from_time=start_ts
        )

        # temp_dict = bybit.kline(symbol=symbol, interval=timeframe, _from=start_ts, limit=limit)
        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_dict):
            symbol_existed = True

        if symbol_existed:
            # extract data and convert to list
            temp_data = [list(i.values())[2:] for i in temp_dict['result']]
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            # NOTE: current implementation does not support inteval of D/W/M/Y
            start_ts = temp_data[len(temp_data) - 1][0] + interval*60

        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        # added start by neo
        if idx == kline_loop_cnt:
            limit = kline_loop_last_limit

        if idx > kline_loop_cnt:
            break

        # added end by neo

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API

        # if idx % 3 == 0:
        #     time.sleep(0.2)



    # convert to data frame
    df = pd.DataFrame(output_data, columns=['TimeStamp', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = [datetime.fromtimestamp(i).strftime('%Y-%m-%d %H:%M:%S.%d')[:-3] for i in df['TimeStamp']]
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    return df


@synchronized
def get_historical_ohlc_data_start_end(symbol, start_int, end_int, past_days=None, interval=None, futures=False):
    try:
        """Returns historcal klines from past for given symbol and interval
        past_days: how many days back one wants to download the data"""
        if not futures:
            # basic
            client_basic = Client("basic_secret_key",
                                  "basic_secret_value")
            client = client_basic
        else:
            # futures
            client_futures = Client("futures_secret_key",
                                    "futures_secret_value")
            client = client_futures

        if not interval:
            interval = '1h'  # default interval 1 hour
        if not past_days:
            past_days = 30  # default past days 30.

        start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).date())
        if end_int:
            end_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_int) + ' days')).date())
        else:
            end_date_str = None
        D = None
        try:
            if exchange_symbol == 'binance_futures':
                if futures:
                    D = pd.DataFrame(
                        client.futures_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))
                else:
                    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))

            elif exchange_symbol == 'bybit_usdt_perp':
                interval = int(interval[:1])
                D = get_historical_klines_pd(symbol, interval, start_date_str, end_date_str, start_int, end_int)
                return D, start_date_str, end_date_str

        except Exception as e:
            time.sleep(0.5)
            # print(e)
            return D, start_date_str, end_date_str

        if D is not None and D.empty:
            return D, start_date_str, end_date_str

        D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'is_best_match']
        D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
        D['symbol'] = symbol
        D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
               'taker_quote_vol']]
        D.rename(columns={
                            "open_date_time": "Date",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                          }, inplace=True)
        new_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        D['Date'] = D['Date'].astype(str)
        D['Open'] = D['Open'].astype(float)
        D['High'] = D['High'].astype(float)
        D['Low'] = D['Low'].astype(float)
        D['Close'] = D['Close'].astype(float)
        D['Volume'] = D['Volume'].astype(float)
        D = D[new_names]
    except Exception as e:
        # print('e in get_historical_ohlc_data_start_end:%s' % e)
        pass
    return D, start_date_str, end_date_str


def fractals_low_loopB(df, loop_count=1):
    for c in range(loop_count):
        i = Indicators(df)
        i.fractals()
        df = i.df
        df = df[~(df[['fractals_low']] == 0).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_low_loopA(df, fcnt=51, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fcnt] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df

def fractals_high_loopA(df, fcnt=5, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_high'] = df['High'].rolling(window, center=True).apply(lambda x: x[fcnt] == max(x), raw=True)
        df = df[~(df[['fractals_high']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high'], axis=1)
    return df

def sma(source, period):
    return pd.Series(source).rolling(period).mean().values

def backtest_trade45(df, symbol, fcnt, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, idx, wavepattern_l, wavepattern_tpsl_l):
    # if not longshort:
    #     print('short short short short short short short short short ')
    #
    t = trade_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w = wavepattern

    # real condititon by fractal index
    real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
    real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    if not real_condititon1:
        if printout:  print('not not not real_condititon1 1111111')
        # print('not not not real_condititon1 1111111')
        return trade_info
    if not real_condititon2:
        if printout:  print('not not not real_condititon2 22222222')
        # print('not not not real_condititon2 2222222')
        return trade_info


    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    entry_price = w.values[7]  # wave4

    df_active = df[w.idx_end + 1:]
    dates = df_active.Date.tolist()
    closes = df_active.Close.tolist()
    trends = df_active.High.tolist() if longshort else df_active.Low.tolist()
    detrends = df_active.Low.tolist() if longshort else df_active.High.tolist()

    fee_maker_maker_percent = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    fee_maker_taker_slippage_percent = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    condi_order_i = []
    position_enter_i = []
    position = False

    # if trade_info:
    #     if trade_info[6]:
    #         h_pre = trade_info[6][-1]
    #         if h_pre.low == w_start_price and h_pre.high == w_end_price:
    #             return trade_info

    if closes:
        for i, close in enumerate(closes):
            c_order = (position is False and close < w_end_price and close > entry_price) \
                if longshort else (position is False and close > w_end_price and close < entry_price)
            c_out_trend_beyond = trends[i] > (w_end_price + o_fibo_value) if longshort else trends[i] < (w_end_price - o_fibo_value)
            c_out_trend = trends[i] > (w_end_price) if longshort else trends[i] < (w_end_price)
            c_out_detrend = detrends[i] < (w_start_price) if longshort else detrends[i] > (w_start_price)

            if intersect_idx:
                trendsline = np.array(trends[0:i+1])
                detrendsline = np.array(detrends[0:i+1])
                entryline = np.array([entry_price for i in range(i+1)])
                # np.sign(...) return -1, 0 or 1
                # np.diff(...) return value difference for (n-1) - n, to obtain intersections
                # np.argwhere(...) remove zeros, preserves turning points only
                detrends_intersect_idx = np.argwhere(np.diff(np.sign(detrendsline - entryline))).flatten()
                trends_intersect_idx = np.argwhere(np.diff(np.sign(trendsline - entryline))).flatten()
                # if detrends_intersect_idx.size > 0:
                #     # if printout: print('detrends_intersect_idx:%s' % detrends_intersect_idx)
                #     pass
                # if trends_intersect_idx.size > 0:
                #     # if printout: print('trends_intersect_idx:%s' % trends_intersect_idx)
                #     pass
                #
                if position is False and detrends_intersect_idx.size > 2:
                    if printout: print('detrends_intersect_out out out:%s' % detrends_intersect_idx)
                    print('detrends_intersect_out out out:%s 22222' % detrends_intersect_idx)
                    return trade_info
                if position is False and trends_intersect_idx.size > 1:
                    if printout: print('trends_intersect_out out out:%s' % detrends_intersect_idx)
                    print('trends_intersect_out out out: 11111 %s' % detrends_intersect_idx)
                    return trade_info

            c_positioning = (position is False and detrends[i] <= entry_price) if longshort else (position is False and detrends[i] >= entry_price)
            c_profit = (position and trends[i] > w_end_price) if longshort else (position and trends[i] < w_end_price)
            c_stoploss = (position and detrends[i] < w_start_price) if longshort else (position and detrends[i] > w_start_price)

            if c_order and not condi_order_i:
                condi_order_i = [dates[i], close]
                if printout: print('===>c_order (not position and c_out_trendc_order and not condi_order_i), ', i, close, condi_order_i)
                # print('c_order (not position and c_out_trendc_order and not condi_order_i), ', i, close, condi_order_i)
            elif position is False and c_out_trend_beyond:
                if printout: print('@@ beyondxxxx @@>c_out_trend (not position and c_out_trend_beyond), ', i, close)
                # print('beyondxxxx @@>c_out_trend (not position and c_out_trend_beyond), ', i, close)
                return trade_info
            # elif position is False and c_out_trend:
            #     if printout: print('@@>c_out_trend (not position and c_out_trend), ', i, close)
            #     return trade_info
            elif position is False and c_out_detrend:
                if printout: print('@@c_out_detrend (not position and c_out_detrend), ', i, close)
                # print('c_out_detrend (not position and c_out_detrend), ', i, close)
                return trade_info
            elif position is False and c_positioning:
                position = True
                position_enter_i = [dates[i], entry_price]
                if printout: print('===>c_positioning (c_positioning), ', i, close, position_enter_i)
                # print('c_positioning (c_positioning), ', i, close, position_enter_i)

            elif position is True:
                if c_profit or c_stoploss:
                    fee_percent = 0
                    pnl_percent = 0
                    trade_inout_i = []
                    if c_stoploss:
                        position_sl_i = [dates[i], w_start_price]
                        pnl_percent = -(abs(entry_price - w_start_price) / entry_price) * leverage
                        fee_percent = fee_maker_taker_slippage_percent
                        trade_count.append(0)
                        trade_inout_i = [position_enter_i, position_sl_i]
                        order_history.append(trade_inout_i)
                        if printout: print('- stoploss, ', i, close, position_enter_i, position_sl_i)

                    if c_profit:
                        position_pf_i = [dates[i], w_end_price]
                        pnl_percent = (abs(w_end_price - entry_price) / entry_price) * leverage
                        fee_percent = fee_maker_maker_percent
                        trade_count.append(1)
                        trade_inout_i = [position_enter_i, position_pf_i]
                        order_history.append(trade_inout_i)
                        if printout: print('+ profit, ', i, close, position_enter_i, position_pf_i)


                    asset_history_pre = asset_history[-1] if asset_history else seed
                    asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                    pnl_history.append(asset_history_pre*pnl_percent)
                    fee_history.append(asset_history_pre*fee_percent)
                    asset_history.append(asset_new)
                    # wavepattern_history.append(wavepattern)

                    winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                    trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history)]
                    stats_history.append(trade_stats)
                    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]

                    wavepattern_tpsl_l.append([idx, wavepattern.dates[0], id(wavepattern), wavepattern])
                    # if printout: print(str(trade_stats))
                    # print(symbol, fcnt, longshort, trade_inout_i[0][0][2:-3], trade_inout_i[0][1], '~', trade_inout_i[1][0][-8:-3], trade_inout_i[1][1], ' | %s' % str([w.values[0], w.values[-1], w.values[7]]), str(trade_stats))
                    print(symbol, fcnt, longshort, trade_inout_i[0][0][2:-3], str(trade_stats))
                    # print(symbol, trade_inut_i[0][0][2:-3], str(trade_stats))

                    if longshort is not None:
                        if plotview:
                            plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]], df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info, title=str(
                                symbol + ' %s '% str(longshort) + str(trade_stats)))

                    return trade_info
    return trade_info


def check_has_same_wavepattern(symbol, fcnt, w_l, wavepattern):
    for wl in reversed(w_l):
        if symbol == wl[0] and fcnt == wl[1]:
            a0 = wl[-1].dates[0]
            b0 = wavepattern.dates[0]
            c0 = wl[-1].values[0]
            d0 = wavepattern.values[0]
            c0 = a0 == b0 and c0 == d0
            if condi_same_date:
                if a0 == b0:
                    return True

            a2 = wl[-1].dates[3]
            b2 = wavepattern.dates[3]
            c2 = wl[-1].values[3]
            d2 = wavepattern.values[3]
            c2 = a2 == b2 and c2 == d2

            a3 = wl[-1].dates[5]
            b3 = wavepattern.dates[5]
            c3 = wl[-1].values[5]
            d3 = wavepattern.values[5]
            c3 = a3 == b3 and c3 == d3

            a4 = wl[-1].dates[7]
            b4 = wavepattern.dates[7]
            c4 = wl[-1].values[7]
            d4 = wavepattern.values[7]
            c4 = a4 == b4 and c4 == d4

            a5 = wl[-1].dates[-1]
            b5 = wavepattern.dates[-1]
            c5 = wl[-1].values[-1]
            d5 = wavepattern.values[-1]
            c5 = a5 == b5 and c5 == d5

            if (c0 and c4) or (c4 and c5) or (c0 and c5) or (c0 and c3) or (c0 and c2 and c4) or (c0 and c2 and c3) or (c0 and c3 and c4):
                return True

            eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(wl[-1].dates))
            eq_values = np.array_equal(np.array(wavepattern.values), np.array(wl[-1].values))
            if eq_dates or eq_values:
                return True

    return False

wavepattern_l = list()

def loopsymbol(symbol, i, trade_info):
    ###################
    ## data settting ##
    ###################
    # 1day:1440m

    # try:

    past_days = 1
    timeunit = 'm'
    bin_size = str(timeframe) + timeunit
    start_int = i
    end_int = start_int - period_interval
    if end_int < 0:
        end_int = None
    df, start_date, end_date = get_historical_ohlc_data_start_end(symbol, start_int=start_int,
                                                                      end_int=end_int, past_days=past_days,
                                                                      interval=bin_size, futures=futures)

    df_all = df
    longshort = None
    wa = WaveAnalyzer(df=df_all, verbose=True)
    wave_options = WaveOptionsGenerator5(up_to=up_to_count)
    idxs = list()
    lows_idxs = list()
    highs_idxs = list()
    df_lows_plot = None
    df_highs_plot = None

    for fc in fcnt:
        if 'long' in type:
            df_lows = fractals_low_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_lows_plot = df_lows[['Date', 'Low']]
            impulse = Impulse('impulse')
            lows_idxs = df_lows.index.tolist()
            idxs = lows_idxs

        if 'short' in type:
            df_highs = fractals_high_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_highs_plot = df_highs[['Date', 'High']]
            downimpulse = DownImpulse('downimpulse')
            highs_idxs = df_highs.index.tolist()
            idxs = highs_idxs

        rules_to_check = list()
        wavepatterns = set()
        # wavepattern_l = []
        wavepattern_tpsl_l = []
        wave_option_plot_l = []
        if ('long' in type) and ('short' in type): idxs = sorted(list(set(lows_idxs) | set(highs_idxs)))
        for i in idxs:
            for wave_opt in wave_options.options_sorted:
                if i in lows_idxs:
                    waves = wa.find_impulsive_wave(idx_start=i, wave_config=wave_opt.values)
                    longshort = True
                    rules_to_check = [impulse]
                elif i in highs_idxs:
                    waves = wa.find_downimpulsive_wave(idx_start=i, wave_config=wave_opt.values)
                    longshort = False
                    rules_to_check = [downimpulse]

                if waves:
                    wavepattern = WavePattern(waves, verbose=True)
                    if (wavepattern.idx_end - wavepattern.idx_start + 1) >= fc / 2:
                        for rule in rules_to_check:
                            if wavepattern.check_rule(rule):
                                if wavepattern in wavepatterns:
                                    continue
                                else:
                                    if not check_has_same_wavepattern(symbol, fc, wavepattern_l, wavepattern):
                                        wavepatterns.add(wavepattern)
                                        wavepattern_l.append([symbol, fc, i, wavepattern.dates[0], id(wavepattern), wavepattern])
                                        wave_option_plot_l.append([
                                            [str(wavepattern.dates[-1])],
                                            [wavepattern.values[-1]],
                                            [str(wave_opt.values)]
                                        ])
                                        trade_info = backtest_trade45(df_all, symbol, fc, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, i, wavepattern_l, wavepattern_tpsl_l)

                                        if printout:
                                            print(f'{rule.name} found: {wave_opt.values}')
                                            print(f'good... {(wavepattern.idx_end - wavepattern.idx_start + 1) }/{fc}found')
                                    else:
                                        if printout:
                                            print(f'{rule.name} found: {wave_opt.values}')
                                            print(f'not good... {(wavepattern.idx_end - wavepattern.idx_start + 1)}/{fc}found')


    # if plotview:
    #     if len(wavepattern_l) > 0:
    #         t = bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(wave_opt.values) + ' ' + str(trade_info[0][-1] if trade_info[0] else [])
    #         plot_pattern_m(df=df_all, wave_pattern=wavepattern_tpsl_l,
    #                        df_lows_plot=df_lows_plot,
    #                        df_highs_plot=df_highs_plot,
    #                        trade_info=trade_info,
    #                        wave_options=wave_option_plot_l, title='tpsl_%s_' % str(fc) + t)
    #     else:
    #         # print(f'not found {wavepattern_l}')
    #         pass


    return trade_info
    # except Exception as e:
    #     print('loopsymbol Exception : %s ' % e)


def single(symbols, i, trade_info, *args):
    for symbol in symbols:
        loopsymbol(symbol, i, trade_info)
    return trade_info


def round_trip(i):
    round_trip = []
    symbols_futures = [
                        # 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                        'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VETUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'OMGUSDT', 'THETAUSDT', 'ENJUSDT', 'MATICUSDT', 'ATOMUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'DOGEUSDT', 'ANKRUSDT', 'MTLUSDT', 'TOMOUSDT', 'DENTUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT', 'NKNUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'BCHUSDT', 'OGNUSDT', 'BTSUSDT', 'COTIUSDT', 'SOLUSDT', 'CTSIUSDT', 'CHRUSDT', 'LENDUSDT', 'STMXUSDT', 'KNCUSDT', 'LRCUSDT', 'COMPUSDT', 'SCUSDT', 'ZENUSDT', 'SNXUSDT', 'DGBUSDT', 'SXPUSDT', 'MKRUSDT', 'STORJUSDT', 'MANAUSDT', 'YFIUSDT', 'BALUSDT', 'BLZUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT', 'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'RUNEUSDT', 'BELUSDT', 'UNIUSDT', 'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'AUDIOUSDT', 'CTKUSDT', 'AKROUSDT', 'AXSUSDT', 'UNFIUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'REEFUSDT', 'CELOUSDT', 'BTCSTUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'ALICEUSDT', 'LINAUSDT', 'TLMUSDT', 'BAKEUSDT', 'ICPUSDT', 'ARUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'ATAUSDT', 'GTCUSDT', 'KEEPUSDT', 'KLAYUSDT', 'C98USDT', 'RAYUSDT', 'DYDXUSDT', 'GALAUSDT', 'ENSUSDT', 'PEOPLEUSDT']

    # symbols_futures = ['FLMUSDT']
    asset_history = [seed]
    trade_count = []
    fee_history = []
    pnl_history = []



    symbols = symbols_futures

    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]
    print(symbols)

    # single
    single(symbols, i, trade_info)
    round_trip.append(trade_info[2][-1])
    print(i, ' | ', trade_info[2][-1], len(trade_count), ' | ', asset_history[-1])
    return round_trip

if __name__ == '__main__':

    print_condition()
    rount_trip_total = []
    start = time.perf_counter()
    if round_trip_flg:
        for i in range(round_trip_count):
            rt = list()
            try:
                start = time.perf_counter()
                # cpucount = multiprocessing.cpu_count() * 1
                # cpucount = multiprocessing.cpu_count()
                cpucount = 1
                print('cpucount:%s' % cpucount)
                pool = multiprocessing.Pool(processes=cpucount)
                rt = pool.map(round_trip, range(period_days_ago, period_days_ago_till, -1 * period_interval))
                pool.close()
                pool.join()
                start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
                end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
                print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
            except Exception as e:
                # print(e)
                pass

            print('============ %s stat.==========' % str(i))
            r = list(map(lambda i: i[0], rt))

            winrate_l = list(map(lambda i: 1 if i[0] > seed else 0, rt))
            meanaverage = round((sum(r)/len(r)), 2)
            roundcount = len(rt)
            winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
            total_gains = (meanaverage - seed)*roundcount
            print('round r: %s' % r)
            print('round winrate_l: %s' % str(winrate_l))
            print('round roundcount: %s' % roundcount)
            print('round winrate: %s' % winrate)
            print('round meanaverage: %s' % str(meanaverage))
            print('round total gains: %s' % str(total_gains))
            print('============ %s End All=========='% str(i))

            rount_trip_total.append([meanaverage, roundcount, winrate, total_gains])
        print_condition()
        for i, v in enumerate(rount_trip_total):
            print(i, v)
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    else:

        symbols = list()

        ## binance symbols
        symbols_futures = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'EOSUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT',
            'TRXUSDT', 'ETCUSDT', 'ICXUSDT', 'VETUSDT', 'LINKUSDT', 'WAVESUSDT', 'BTTUSDT', 'HOTUSDT', 'ZILUSDT',
            'ZRXUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'IOSTUSDT', 'CELRUSDT', 'DASHUSDT', 'OMGUSDT', 'THETAUSDT',
            'ENJUSDT', 'MATICUSDT', 'ATOMUSDT', 'ONEUSDT', 'FTMUSDT', 'ALGOUSDT', 'DOGEUSDT', 'ANKRUSDT', 'MTLUSDT',
            'TOMOUSDT', 'DENTUSDT', 'CVCUSDT', 'CHZUSDT', 'BANDUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HBARUSDT',
            'NKNUSDT', 'KAVAUSDT', 'ARPAUSDT', 'IOTXUSDT', 'RLCUSDT', 'BCHUSDT', 'OGNUSDT', 'BTSUSDT', 'COTIUSDT',
            'SOLUSDT', 'CTSIUSDT', 'CHRUSDT', 'LENDUSDT', 'STMXUSDT', 'KNCUSDT', 'LRCUSDT', 'COMPUSDT', 'SCUSDT',
            'ZENUSDT', 'SNXUSDT', 'DGBUSDT', 'SXPUSDT', 'MKRUSDT', 'STORJUSDT', 'MANAUSDT', 'YFIUSDT', 'BALUSDT',
            'BLZUSDT', 'SRMUSDT', 'ANTUSDT', 'CRVUSDT', 'SANDUSDT', 'OCEANUSDT', 'DOTUSDT', 'LUNAUSDT', 'RSRUSDT',
            'TRBUSDT', 'BZRXUSDT', 'SUSHIUSDT', 'YFIIUSDT', 'KSMUSDT', 'EGLDUSDT', 'RUNEUSDT', 'BELUSDT', 'UNIUSDT',
            'AVAXUSDT', 'HNTUSDT', 'FLMUSDT', 'ALPHAUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'AUDIOUSDT', 'CTKUSDT',
            'AKROUSDT', 'AXSUSDT', 'UNFIUSDT', 'XEMUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'REEFUSDT', 'CELOUSDT',
            'BTCSTUSDT', 'LITUSDT', 'SFPUSDT', 'DODOUSDT', 'ALICEUSDT', 'LINAUSDT', 'TLMUSDT', 'BAKEUSDT', 'ICPUSDT',
            'ARUSDT', 'MASKUSDT', 'LPTUSDT', 'NUUSDT', 'ATAUSDT', 'GTCUSDT', 'KEEPUSDT', 'KLAYUSDT', 'C98USDT',
            'RAYUSDT', 'DYDXUSDT', 'GALAUSDT', 'ENSUSDT', 'PEOPLEUSDT']


        ## bybit symbols
        symbols_bybit = []
        symbols_info = session.query_symbol()
        if symbols_info['ret_msg'] == 'OK':
            symbol_result = symbols_info['result']
            for sym in symbol_result:
                quote_currenty = sym['quote_currency']
                if quote_currenty == 'USDT':
                    symbol_name = sym['name']
                    symbols.append(symbol_name)
            symbols_bybit = symbols

        if symbol_duplicated:
            symbols = [x for x in symbols_futures if x in symbols_bybit]
        elif exchange == 'bybit_usdt_perp':
            symbols = symbols_bybit
        elif exchange == 'binance_futures':
            symbols = symbols_futures


        if symbol_last:
            symbols = symbols[symbol_last:]
        if symbol_length:
            symbols = symbols[:symbol_length]
        if symbol_random:
            symbols = random.sample(symbols, len(symbols))

        print(len(symbols), symbols)

        stats_history = []
        order_history = []
        asset_history = []
        trade_count = []
        fee_history = []
        pnl_history = []
        wavepattern_history = []
        trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]
        r = range(period_days_ago, period_days_ago_till, -1 * period_interval)
        for i in r:
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            single(symbols, i, trade_info)
            if trade_info[2]:
                print(str(i)+'/'+str(len(r)), ' now asset: ', trade_info[2][-1], ' | ', len(trade_count), ' | pre seed: ', asset_history_pre)
            else:
                print(str(i)+'/'+str(len(r)), ' now asset: ', seed, ' | ', len(trade_count), ' | pre seed: ', seed)

        print('============ %s stat.==========' % str(i))
        winrate_l = list(map(lambda i: 1 if i > 0 else 0, pnl_history))
        meanaverage = round((sum(asset_history)/len(asset_history)), 2)
        roundcount = len(trade_count)
        winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
        print('round r: %s' % roundcount)
        print('round winrate_l: %s' % str(winrate_l))
        print('round roundcount: %s' % roundcount)
        print('round winrate: %s' % winrate)
        print('round meanaverage: %s' % str(meanaverage))
        print('round total gains: %s' % str(trade_info[-2][-1] if trade_info[2] else 0))
        print('============ %s End All=========='% str(i))
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    print("good luck done!!")