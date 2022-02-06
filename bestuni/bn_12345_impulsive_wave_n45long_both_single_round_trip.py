from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
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

import json
with open('config.json', 'r') as f:
    config = json.load(f)

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

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fractal_count = config['default']['fractal_count']
loop_count = config['default']['loop_count']


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


def print_condition():
    print('-------------------------------')
    print('futures:%s' % str(futures))
    print('rule:%s' % str(rule))
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
    # print('period_days_ago_till: %s' % period_days_ago_till)
    print('period_interval: %s' % period_interval)
    print('round_trip_count: %s' % round_trip_count)
    print('compounding: %s' % compounding)
    print('fractal_count: %s' % fractal_count)
    print('loop_count: %s' % loop_count)

    print('symbol_random: %s' % symbol_random)
    print('symbol_length: %s' % symbol_length)

    # print('timeframe: %s' % timeframe)
    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    print('period: %s ~ %s' % (start_dt, end_dt))
    print('up_to_count: %s' % up_to_count)
    # print('long: %s' % long)
    print('o_fibo: %s' % o_fibo)
    print('h_fibo: %s' % h_fibo)
    print('l_fibo: %s' % l_fibo)

    print('plotview: %s' % plotview)
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

@synchronized
def get_historical_ohlc_data_start_end(symbol, start_str, end_str, past_days=None, interval=None, futures=False):
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

        start_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_str) + ' days')).date())
        if end_str:
            end_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_str) + ' days')).date())
        else:
            end_str = None
        # print(start_str, end_str)
        D = None
        try:
            if futures:
                D = pd.DataFrame(
                    client.futures_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str, interval=interval))
            else:
                D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, end_str=end_str, interval=interval))
            pass
        except Exception as e:
            time.sleep(0.5)
            # print(e)
            return D, start_str, end_str

        if D is not None and D.empty:
            return D, start_str, end_str

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
        D['Open'] = D['Open'].astype(float)
        D['High'] = D['High'].astype(float)
        D['Low'] = D['Low'].astype(float)
        D['Close'] = D['Close'].astype(float)
        D['Volume'] = D['Volume'].astype(float)
        D = D[new_names]
    except Exception as e:
        # print('e in get_historical_ohlc_data_start_end:%s' % e)
        pass
    return D, start_str, end_str



def fractals_low_loopB(df, loop_count=1):
    for c in range(loop_count):
        i = Indicators(df)
        i.fractals()
        df = i.df
        df = df[~(df[['fractals_low']] == 0).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df


def fractals_low_loopA(df, fractal_count=51, loop_count=1):
    for c in range(loop_count):
        window = 2 * fractal_count + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fractal_count] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df


def real_check(df_all, posi_start):
    if posi_start:
        c1 = df_all['Date'] < posi_start
        df_real = df_all[c1]
        df_lows_real = fractals_low_loopA(df_real, fractal_count=51, loop_count=1)
        mins_real = df_lows_real.index.tolist()
        # start_wave_date = i.iloc[i]['Date']
        if len(mins_real) > 0:
            if i not in mins_real:
                # print('리스트에 값이 없습니다.')
                return False
            else:
                # print('리스트에 값이 있습니다.@@@@@@@@@@@@@@@@@@@@@@@@@@')
                return True


def backtest_trade45(df, wavepattern, symbol, trade_info):
    t = trade_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w = wavepattern
    # w_high_price = w.high
    # w_low_price = w.low
    # w_start_time = w.dates[0]`1
    # w_end_time = w.dates[-1]

    w_start_price = w.values[0]
    w_end_price = w.values[-1]
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    entry_price = w.waves['wave4'].low if long else w.waves['wave4'].high

    df_active = df[w.idx_end + 1:]
    dates = df_active.Date.tolist()
    closes = df_active.Close.tolist()
    trends = df_active.High.tolist() if long else df_active.Low.tolist()
    detrends = df_active.Low.tolist() if long else df_active.High.tolist()

    fee_maker_maker_percent = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    fee_maker_taker_slippage_percent = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
    condi_order_i = []
    position_enter_i = []
    position_pf_i = []
    position_sl_i = []
    position = False

    # if trade_info:
    #     if trade_info[6]:
    #         h_pre = trade_info[6][-1]
    #         if h_pre.low == w_start_price and h_pre.high == w_end_price:
    #             return trade_info

    if closes:
        for i, close in enumerate(closes):
            c_order = (position is False and close < w_end_price and close > entry_price) \
                if long else (position is False and close > w_end_price and close < entry_price)
            c_out_trend = trends[i] > (w_end_price + o_fibo_value) if long else trends[i] < (w_end_price - o_fibo_value)
            c_out_detrend = detrends[i] < (w_start_price) if long else detrends[i] > (w_start_price)
            c_positioning = (position is False and detrends[i] <= entry_price) if long else (position is False and detrends[i] >= entry_price)
            c_profit = (position and trends[i] >= w_end_price) if long else (position and trends[i] <= w_end_price)
            c_stoploss = (position and detrends[i] <= w_start_price) if long else (position and detrends[i] >= w_start_price)

            if c_order and not condi_order_i:
                condi_order_i = [dates[i], close]
            elif not position and c_out_trend:
                return trade_info
            elif not position and c_out_detrend:
                return trade_info
            elif c_positioning:
                position = True
                position_enter_i = [dates[i], entry_price]
            elif position:
                if c_profit or c_stoploss:
                    fee_percent = 0
                    pnl_percent = 0
                    if c_profit:
                        position_pf_i = [dates[i], w_end_price]
                        pnl_percent = (abs(w_end_price - entry_price) / entry_price ) * leverage
                        fee_percent = fee_maker_maker_percent
                        trade_count.append(1)

                    if c_stoploss:
                        position_sl_i = [dates[i], w_start_price]
                        pnl_percent = -(abs(entry_price - w_start_price) / entry_price) * leverage
                        fee_percent = fee_maker_taker_slippage_percent
                        trade_count.append(0)

                    asset_history_pre = asset_history[-1] if asset_history else seed
                    asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                    pnl_history.append(asset_history_pre*pnl_percent)
                    fee_history.append(asset_history_pre*fee_percent)
                    asset_history.append(asset_new)
                    # wavepattern_history.append(wavepattern)

                    winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                    trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history)]
                    stats_history.append(trade_stats)
                    order_history.append([condi_order_i, position_enter_i, position_pf_i, position_sl_i])
                    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]
                    if printout: print(str(trade_stats))
                    return trade_info
    return trade_info


def check_has_same_wavepattern(w_l, wavepattern):
    for i in w_l:
        eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(i[-1].dates))
        eq_values = np.array_equal(np.array(wavepattern.values), np.array(i[-1].values))
        if eq_dates and eq_values:
            return True
    return False


def loopsymbol(symbol, i, trade_info):
    ###################
    ## data settting ##
    ###################
    # 1day:1440m

    try:
        past_days = 1
        # timeframe = 1  # 1, 5, 15
        timeunit = 'm'
        bin_size = str(timeframe) + timeunit
        # stick_cnt_per_day = past_days * 1440 / timeframe if timeunit == 'm' else 0
        # divide = 4
        start_str = i
        end_str = start_str - period_interval
        if end_str < 0:
            end_str = None
        df, start_date, end_date = get_historical_ohlc_data_start_end(symbol, start_str=start_str,
                                                                          end_str=end_str, past_days=past_days,
                                                                          interval=bin_size, futures=futures)
        df_all = df
        df_lows = fractals_low_loopA(df_all, fractal_count=fractal_count, loop_count=loop_count)
        df_lows_plot = df_lows[['Date', 'Low']]
        wa = WaveAnalyzer(df=df_all, verbose=True)
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)  # generates WaveOptions up to [15, 15, 15, 15, 15]
        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        wavepatterns_up = set()

        mins = df_lows.index.tolist()
        wavepattern_up_l = []
        wave_option_plot_l = []
        for i in mins:
            for new_option_impulse in wave_options_impulse.options_sorted:
                waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
                if waves_up:
                    wavepattern = WavePattern(waves_up, verbose=True)
                    if (wavepattern.idx_end - wavepattern.idx_start + 1) >= fractal_count / 2:
                        for rule in rules_to_check:
                            if wavepattern.check_rule(rule):
                                if wavepattern in wavepatterns_up:
                                    continue
                                else:
                                    if not check_has_same_wavepattern(wavepattern_up_l, wavepattern):

                                        wavepatterns_up.add(wavepattern)
                                        # print(f'{rule.name} found: {new_option_impulse.values}')
                                        # print(f'good... {(wavepattern.idx_end - wavepattern.idx_start + 1) }/{fractal_count}found')

                                        # plot_pattern(df=df, wave_pattern=wavepattern, title=str(
                                        # symbol + '_' + str(i) + ':' + rule.name + ' ' + str(
                                        #     new_option_impulse.values)))
                                        # plot_pattern_m(df=df_all, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]], df_plot=df_lows_plot, title=str(
                                        #     symbol + '_II_' + '_' + str(i) + ':' + rule.name + ' ' + str(
                                        #         new_option_impulse.values)))
                                        wavepattern_up_l.append([i, wavepattern.dates[0], id(wavepattern), wavepattern])
                                        wave_option_plot_l.append([
                                            [str(wavepattern.waves['wave5'].dates[1])],
                                            [wavepattern.waves['wave5'].high],
                                            [str(new_option_impulse.values)]
                                        ])
                                        trade_info = backtest_trade45(df_all, wavepattern, symbol, trade_info)
                                    # else:
                                    #     # print(f'{rule.name} found: {new_option_impulse.values}')
                                    #     # print(f'not good... {(wavepattern.idx_end - wavepattern.idx_start + 1)}/{fractal_count}found')
                                    #     pass


        if plotview:
            if len(wavepattern_up_l) > 0:
                t = bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(new_option_impulse.values) + ' ' + str(trade_info[0][-1] if trade_info[0] else [])
                plot_pattern_m(df=df_all, wave_pattern=wavepattern_up_l,
                               df_plot=df_lows_plot,
                               trade_info=trade_info,
                               wave_options=wave_option_plot_l, title=t)
        return trade_info
    except Exception as e:
        print('loopsymbol Exception : %s ' % e)


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
        symbols_futures = [
            # 'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
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

        # symbols_futures = ['FLMUSDT']
        stats_history = []
        order_history = []
        asset_history = []
        trade_count = []
        fee_history = []
        pnl_history = []
        wavepattern_history = []
        trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]
        symbols = symbols_futures
        if symbol_random:
            symbols = random.sample(symbols, len(symbols))
        if symbol_length:
            symbols = symbols[:symbol_length]
        print(symbols)

        for i in range(period_days_ago, period_days_ago_till, -1 * period_interval):
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            single(symbols, i, trade_info)
            print(i, ' now asset: ', trade_info[2][-1], ' | ', len(trade_count), ' | pre seed: ', asset_history_pre)

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
        print('round total gains: %s' % str(trade_info[-2][-1]))
        print('============ %s End All=========='% str(i))
        print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    print("good luck done!!")