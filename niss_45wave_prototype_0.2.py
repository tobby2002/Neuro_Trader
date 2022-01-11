from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern, plot_pattern_m
from numba import njit

import pandas as pd
import numpy as np
import random
from pytz import UTC
from datetime import datetime, timezone
from datetime import date, timedelta
import json
import time
import os
from tapy import Indicators
from tqdm import tqdm

with open('config.json', 'r') as f:
    config = json.load(f)


def check_has_same_wavepattern_df(df_wave, wavepattern_up):
    l = df_wave.wave.tolist()
    if l:
        for i in l:
            eq_dates = np.array_equal(np.array(wavepattern_up.dates), np.array(i[1]))
            eq_values = np.array_equal(np.array(wavepattern_up.values), np.array(i[2]))
            if eq_dates and eq_values:
                return True
    return False


def get_status_and_candle_trajectory(df_candle, nowpoint, wtime, w4, low, high, position, status):

    if status == 0:
        max = np.max(df_candle[wtime:nowpoint].High.tolist())
        if max >= high:
            return 2
        min = np.min(df_candle[wtime:nowpoint].Low.tolist())
        if min <= w4:
            return 1
        return 0

def agent_filter_and_update_wave_df(df_wave, symbol,  bin_size, bin_time, nowpoint, df_candle):
    c1 = df_wave['symbol'] == symbol
    c2 = df_wave['bin_size'] == bin_size
    c3 = df_wave['bin_time'] == bin_time
    c4 = df_wave['time'] < nowpoint
    c5 = df_wave['position'] == 0
    c6 = df_wave['status'] == 0
    df_wave = df_wave[c1 & c2 & c3 & c4
                      & c5 & c6]

    df_candle = df_candle[df_candle['Date'] <= nowpoint]
    df_candle = df_candle.set_index(['Date'])

    if not df_wave.empty:
        df_wave.loc[:, ('status')] = df_wave.apply(lambda x: get_status_and_candle_trajectory(
            df_candle, nowpoint, x['time'], x['low'], x['w4'], x['high'], x['position'], x['status']), axis=1)
    return df_wave


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


def print_condition():
    print('-------------------------------')
    print('futures:%s' % str(futures))
    print('rule:%s' % str(rule))
    print('leverage:%s' % str(leverage))
    print('seed:%s' % str(seed))
    print('fee:%s%%' % str(fee * 100))
    print('fee_maker:%s%%' % str(fee_maker * 100))
    print('fee_taker:%s%%' % str(fee_taker * 100))
    print('fee_slippage:%s%%' % str(round(fee_slippage * 100, 4)))
    if futures:
        fee_high = (
                           fee_taker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        fee_low = (
                          fee_taker + fee_maker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_high+fee_low)*100/2 + slippage:%s%%' % round(
            (float(fee_high) + float(fee_low)) * 100 / 2 + fee_slippage * 100, 4))

    else:
        fee_high = (
                           fee_taker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        fee_low = (
                          fee_taker + fee_maker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        print('(fee_high+fee_low)*100/2 + slippage:%s%%' % round(
            (float(fee_high) + float(fee_low)) * 100 / 2 + fee_slippage * 100, 4))

    print('timeframe: %s' % timeframe)
    print('period_days_ago: %s' % period_days_ago)
    # print('period_days_ago_till: %s' % period_days_ago_till)
    print('period_interval: %s' % period_interval)
    print('round_trip_count: %s' % round_trip_count)
    print('compounding: %s' % compounding)

    print('symbol_random: %s' % symbol_random)
    print('symbol_length: %s' % symbol_length)

    # print('timeframe: %s' % timeframe)
    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    print('period: %s ~ %s' % (start_dt, end_dt))
    print('up_to_count: %s' % up_to_count)
    # print('long: %s' % long)
    print('h_fibo: %s' % h_fibo)
    print('l_fibo: %s' % l_fibo)
    print('-------------------------------')

################
# wave analyzer
################
def wave_analyzer(dm, exchange, symbols, bin_size, bin_time, up_to_count, startpoint, nowpoint, window, truncate=False):
    pid = os.getpid()
    print('wave_analyzer start pid: %s' % pid)
    conn = dm.get_conn()
    if truncate:
        dm.truncate_tablename(conn, 'wave')
        print('truncated wave')
    for symbol in symbols:
        table = exchange + '_' + symbol.lower()
        try:
            df_candle = dm.load_df_ohlc_startpoint_endpoint_window(conn,
                                table, bin_size, bin_time, startpoint, nowpoint, window)
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
        df_lows = fractals_low_loop(df_all)
        df_lows = fractals_low_loop(df_lows)
        df_lows_plot = df_lows[['Date', 'Low']]
        wa = WaveAnalyzer(df=df_all, verbose=False)
        up_to_count = up_to_count if up_to_count else 1
        wave_options_impulse = WaveOptionsGenerator5(up_to=up_to_count)

        impulse = Impulse('impulse')
        rules_to_check = [impulse]
        wavepattern_up_l = []
        wave_option_plot_l = []
        wavepatterns_up = set()
        mins = df_lows.index.tolist()
        for i in mins:
            for new_option_impulse in wave_options_impulse.options_sorted:
                waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
                if waves_up:
                    wavepattern_up = WavePattern(waves_up, verbose=True)
                    for rule in rules_to_check:
                        if wavepattern_up.check_rule(rule):
                            if wavepattern_up in wavepatterns_up:
                                # print('################################')
                                # print('same wavepattern id:%s' % id(wavepattern_up))
                                # print('################################')
                                continue
                            else:
                                wavepatterns_up.add(wavepattern_up)
                                wavepattern_up_to_list = [rule.name, wavepattern_up.dates, wavepattern_up.values,
                                                          wavepattern_up.labels]
                                if not check_has_same_wavepattern_df(df_wave, wavepattern_up):
                                    w_info = [
                                              id(wavepattern_up),
                                              symbol,
                                              bin_size,
                                              bin_time,
                                              rule.name,
                                              wavepattern_up.dates[0],
                                              wavepattern_up.low,
                                              wavepattern_up.values[1],
                                              wavepattern_up.values[3],
                                              wavepattern_up.values[5],
                                              wavepattern_up.values[7],
                                              wavepattern_up.high,
                                              str(wavepattern_up_to_list),
                                              0,
                                              0
                                              ]

                                    # insert into a wave
                                    dm.save_wave(w_info, conn)
                                    print('len(df_wave):%s' % len(df_wave))
                                    print(f'{rule.name} found: {new_option_impulse.values}')
                                    wavepattern_up_l.append(wavepattern_up)
                                    wave_option_plot_l.append([
                                        [str(wavepattern_up.waves['wave5'].dates[1])],
                                        [wavepattern_up.waves['wave5'].high],
                                        [str(new_option_impulse.values)]
                                    ])

        if plotview:
            if wavepattern_up_l:
                t = str(symbol + '_ALL_' + bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(new_option_impulse.values) + ' ')
                plot_pattern_m(df=df_all, wave_pattern=wavepattern_up_l,
                               df_plot=df_lows_plot,
                               wave_options=wave_option_plot_l, title=t)
                print(t)
    conn.commit()
    conn.close()


################
# wave manager
################
def wave_manager(dm, exchange, symbols, bin_size, bin_time, type, startpoint=None, nowpoint=None,  window=720, status_init=False):
    try:
        conn = dm.get_conn()
        if status_init:
            dm.update_wave_init(conn)

        df_wave = dm.load_df_wave_startpoint_endpoint_window(conn, bin_size, bin_time, type,
                                                             startpoint=startpoint, endpoint=nowpoint, window=window)

    except Exception as e:
        print('load_df_wave_startpoint_endpoint_window in wave_manager, e:%s' % e)
        return None

    if nowpoint is None:
        nowpoint = str(datetime.now().astimezone(UTC))
        print('nowpoint is None, set nowpoint')
        return None

    wave_standby_l = list()
    if not df_wave.empty:
        for symbol in symbols:
            table = exchange + '_' + symbol.lower()
            try:
                df_candle = dm.load_df_ohlc_startpoint_endpoint_window(conn, table, bin_size, bin_time,
                                                                       startpoint=startpoint, endpoint=nowpoint, window=window)
            except Exception as e:
                print('df_candle in wave_manager load_df:%s' % e)
                continue
            standby_df = agent_filter_and_update_wave_df(df_wave, symbol,  bin_size, bin_time, nowpoint, df_candle)
            if not standby_df.empty:
                wave_standby_l.append(standby_df)
                dm.update_wave(standby_df, -1, dm.get_conn())  # -1 -> status, -2 -> position
    else:
        print('df_wave empty')
    conn.commit()
    conn.close()
    return wave_standby_l


if __name__ == "__main__":
    try:
        from datamgr import DataManagerfeeder
    except:
        from ..xtemp.test.datamgr import DataManagerfeeder

    # symbols_futures = ['FLMUSDT']
    asset_total = [seed]
    trade_count = []
    fee_history = []
    pnl_history = []
    trade_history = [None, None, asset_total, trade_count, fee_history, pnl_history, None]
    # df_wave = pd.DataFrame([],
    #                        columns=['id', 'symbol', 'bin_size', 'bin_time', 'type', 'time', 'low', 'w1', 'w2', 'w3', 'w4', 'high', 'wave',
    #                                 'position', 'status'])
    # if symbol_random:
    #     symbols = random.sample(symbols, len(symbols))
    # if symbol_length:
    #     symbols = symbols[:symbol_length]
    symbols = symbols[:symbol_length]
    # print(symbols)

    dm = DataManagerfeeder()
    ######################### create table and data #######################
    # symbols_d = data_init(dm, symbols, exchange, bin_size, start_str, end_str, None, futures)
    # print(symbols_d)

    # create create_wave
    # conn = dm.get_conn()
    # dm.create_wave(conn)
    # conn.commit()
    # conn.close()

    # conn = dm.get_conn()
    # df_wave = dm.load_df_wave(conn)
    # conn.commit()
    # conn.close()


    # get symbols_d
    # symbols_d = data_symbols_d(dm, symbols, exchange, bin_size)

    #### 0 (ohlc scheduler)

    ##################################################
    #### 1 (wave_analyzer scheduler)
    ##################################################
    # window = 1440 / 2
    # # startpoint = '2021-12-24'
    # startpoint = None
    # nowpoint = '2021-12-25'
    # truncate = True
    # bin_size = 1
    # bin_time = 'm'
    #
    # print('1 (wave_analyzer scheduler)')
    # print('startpoint:%s' % startpoint)
    # print('truncate:%s' % truncate)
    # start = time.perf_counter()
    # wave_analyzer(dm, exchange, symbols, bin_size, bin_time, up_to_count=up_to_count,
    #                                        startpoint=startpoint, nowpoint=nowpoint, window=window, truncate=truncate)
    # print(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')




    ##################################################
    #### 2 (wave_manager conditioner)
    ##################################################
    sdate = date(2021, 12, 24)  # start date
    sdate = '2021-12-24 14:20:00'
    edate = date(2022, 12, 25)  # end date
    symbols_d = dict()
    window = 1440 / 2
    balance = 100
    date_l = pd.date_range(sdate, edate-timedelta(days=1), freq='1min')
    print('date_l sdate:%s' % sdate)
    print('date_l edate:%s' % edate)
    # print(date_l.tolist())
    startpoint = '2021-12-24'
    nowpoint = '2021-12-25'
    print('startpoint:%s' % startpoint)
    print('nowpoint:%s' % nowpoint)
    timeframe = '1m'
    bin_size = 1
    bin_time = 'm'
    type = 'impulse'
    entry_condition = True
    status_init = True

    wave_standby_l = None
    i = 0
    for nowpoint in date_l.tolist():
        if i != 0:
            status_init = False
        start = time.perf_counter()
        wave_standby_l = wave_manager(dm, exchange, symbols, bin_size, bin_time, type, startpoint=str(startpoint), nowpoint=str(nowpoint), window=window, status_init=status_init)
        i += 1
        print(f'Finished wave_managerI in {round(time.perf_counter() - start, 2)} second(s)')

        #### 3 (status_manager)
        if wave_standby_l:
            start = time.perf_counter()
            for df in wave_standby_l:
                w = df.values.tolist()
                if w[-1][-1] == 2:
                    dm.update_wave_key_value(w[0][0], 'position', 1, dm.get_conn())
            print(f'Finished wave_managerII in {round(time.perf_counter() - start, 2)} second(s)')

    print('ALL DONE!!!')


    #### 4 (position_manager)

    #### 5 (balance_manager)


