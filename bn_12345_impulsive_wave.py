from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, LeadingDiagonal, Correction
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern
import pandas as pd
import numpy as np
import math
from binance.client import Client
# matplotlib.use('TkAgg')
import datetime as dt
from tapy import Indicators
import time
import multiprocessing
from functools import partial

client = Client("knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq", "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE")
client = Client()


def get_historical_ohlc_data(symbol, past_days=None, interval=None):
    """Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data"""

    if not interval:
        interval = '1h'  # default interval 1 hour
    if not past_days:
        past_days = 30  # default past days 30.

    start_str = str((pd.to_datetime('today') - pd.Timedelta(str(past_days) + ' days')).date())

    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_str, interval=interval))
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
    return D

# 1day:1440m
past_days = 1
interval = 1  # 1, 5, 15
timeunit = 'm'
interval_str = str(interval) + timeunit
stick_cnt_per_day = past_days * 1440 / interval if timeunit == 'm' else 0
divide = 5
period_unit = math.floor(stick_cnt_per_day/divide)
print('past_days:%s interval_str:%s period_unit:%s (total_unit:%s)' % (past_days, interval_str, period_unit, stick_cnt_per_day))


symbols = [
            'BTCUSDT',
            'ETHUSDT',
            'ALICEUSDT',
            'GTCUSDT',
            # 'TLMUSDT',
            # 'EGLDUSDT',
            # 'FTMUSDT',
            # 'AXSUSDT',
            # 'NUUSDT',
            # 'LITUSDT',
           ]
print(symbols)


def loopoptions(symbol, wa, new_option_impulse, idx_start, df, rules_to_check, wavepatterns_up):
    waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
    if waves_up:
        wavepattern_up = WavePattern(waves_up, verbose=True)
        for rule in rules_to_check:
            if wavepattern_up.check_rule(rule):
                if wavepattern_up in wavepatterns_up:
                    continue
                else:
                    wavepatterns_up.add(wavepattern_up)
                    print(f'{rule.name} found: {new_option_impulse.values}')
                    plot_pattern(df=df, wave_pattern=wavepattern_up,
                                 title=str(symbol + '_' + interval_str + ':' + str(new_option_impulse)))

from concurrent import futures

def loopsymbol(symbol):
    print('\n\n symbol %s start' % (symbol))
    df = get_historical_ohlc_data(symbol, past_days=past_days, interval=interval_str)

    i = Indicators(df)
    # i.accelerator_oscillator(column_name='AC')
    # i.atr()
    # i.sma()
    # i.mfi()
    # i.ichimoku_kinko_hyo()
    i.fractals()
    df = i.df
    # df.tail()

    # df = pd.read_csv(r'data\btc-usd_1d.csv')
    # df = pd.read_csv(r'btc-usd_1d.csv')

    idx_start_list = df.index[df['fractals_low'] == 1.00000].tolist()


    idx_start = np.argmin(np.array(list(df['Low'])))
    idx_start_list = [idx_start]
    # wa = WaveAnalyzer(df=df, verbose=False)
    # start = time.time()  # 시작 시간 저장
    wa = WaveAnalyzer(df=df, verbose=True)
    wave_options_impulse = WaveOptionsGenerator5(up_to=15)  # generates WaveOptions up to [15, 15, 15, 15, 15]
    # print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    impulse = Impulse('impulse')
    leading_diagonal = LeadingDiagonal('leading diagonal')
    correction = Correction('correction')

    rules_to_check = [impulse]
    # rules_to_check = [impulse, leading_diagonal]
    # rules_to_check = [impulse, leading_diagonal, correction]

    print(f'Start at idx: {idx_start}')
    print(f"will run up to {wave_options_impulse.number / 1e6}M combinations.")

    # set up a set to store already found wave counts
    # it can be the case, that 2 WaveOptions lead to the same WavePattern.
    # This can be seen in a chart, where for example we try to skip more maxima as there are. In such a case
    # e.g. [1,2,3,4,5] and [1,2,3,4,10] will lead to the same WavePattern (has same sub-wave structure, same begin / end,
    # same high / low etc.
    # If we find the same WavePattern, we skip and do not plot it

    wavepatterns_up = set()

    # loop over all combinations of wave options [i,j,k,l,m] for impulsive waves sorted from small, e.g.  [0,1,...] to
    # large e.g. [3,2, ...]
    print(f'idxs: {idx_start_list}')
    for i in idx_start_list:
        print(f'idxs: {symbol} {idx_start_list}')
        print(f'Start at idx: {symbol} {i}')
        for new_option_impulse in wave_options_impulse.options_sorted:
            # print(f'Start at idx: {i}: {new_option_impulse}')

            # loopoptions(symbol, wa, new_option_impulse, idx_start, df, rules_to_check, wavepatterns_up)
            # waves_up = wa.find_impulsive_wave(idx_start=idx_start, wave_config=new_option_impulse.values)
            waves_up = wa.find_impulsive_wave(idx_start=i, wave_config=new_option_impulse.values)
            if waves_up:
                wavepattern_up = WavePattern(waves_up, verbose=True)
                for rule in rules_to_check:
                    if wavepattern_up.check_rule(rule):
                        if wavepattern_up in wavepatterns_up:
                            continue
                        else:
                            wavepatterns_up.add(wavepattern_up)
                            print(f'{rule.name} found: {new_option_impulse.values}')
                            plot_pattern(df=df, wave_pattern=wavepattern_up, title=str(symbol +'_'+ interval_str +':'+ rule.name +' '+ str(new_option_impulse.values)))


            # print(f'End at idx: {i}')
            # print(f'End at idx: {symbol} {i}')
            pass
        print('===============================')
        print('impulsive wave end: %s _ index: %s' % (symbol, i))
        print('===============================')

# for no, symbol in enumerate(symbols):
#     loopsymbol(symbol)

if __name__ == '__main__':

    start = time.perf_counter()
    cpucount = multiprocessing.cpu_count() * 2
    print('cpucount:%s' % cpucount)
    pool = multiprocessing.Pool(processes=cpucount)
    rt = pool.map(loopsymbol, symbols)
    pool.close()
    pool.join()
    print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
    print('============End All==========')
